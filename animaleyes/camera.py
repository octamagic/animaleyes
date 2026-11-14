"""Frame ingestion from a Kasa camera through an ffmpeg subprocess.

ffmpeg decodes the stream and emits JPEGs at ~2 fps on stdout; we split them on JPEG
markers and keep the most recent ones in a ring buffer. No OpenCV, no video decoding in
Python. If ffmpeg dies or the camera goes away we restart it with backoff.
"""

from __future__ import annotations

import base64
import concurrent.futures
import ipaddress
import logging
import select
import socket
import subprocess
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from urllib.parse import urlsplit

from .config import Secrets

log = logging.getLogger(__name__)

JPEG_START = b"\xff\xd8"
JPEG_END = b"\xff\xd9"
FRAME_WIDTH = 640  # keeps LLM image tokens low; the bowl area stays legible
CAMERA_STALL_S = 20.0  # no bytes from ffmpeg for this long = stalled stream; reconnect


@dataclass(frozen=True)
class Frame:
    jpeg: bytes
    at: datetime


class FrameBuffer:
    """Thread-safe ring buffer of the most recent frames."""

    def __init__(self, size: int = 12):
        self._frames: deque[Frame] = deque(maxlen=size)
        self._lock = threading.Lock()

    def push(self, frame: Frame) -> None:
        with self._lock:
            self._frames.append(frame)

    def latest(self, n: int = 1) -> list[Frame]:
        with self._lock:
            return list(self._frames)[-n:]

    def last_frame_at(self) -> datetime | None:
        with self._lock:
            return self._frames[-1].at if self._frames else None


def kasa_legacy_password(password: str) -> str:
    """Kasa cameras want base64(account password) as the basic-auth password.

    Matches go2rtc's kasa source (`secret1` -> `c2VjcmV0MQ==`), tested there on EC71.
    """
    return base64.b64encode(password.encode("utf-8")).decode("ascii")


def arp_lookup(mac: str, arp_table: Path = Path("/proc/net/arp")) -> str | None:
    """IPv4 address the kernel currently maps to `mac`, if any."""
    for line in arp_table.read_text().splitlines()[1:]:
        fields = line.split()
        # IP address, HW type, Flags, HW address, Mask, Device. Flags 0x0 is an incomplete entry.
        if len(fields) >= 4 and fields[3].lower() == mac.lower() and fields[2] != "0x0":
            return fields[0]
    return None


def local_subnet() -> ipaddress.IPv4Network:
    """The /24 of the LAN address this box routes out of. POC: assumes a /24 home network."""
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.connect(("192.0.2.1", 9))  # UDP connect sends nothing; it only picks a route
        return ipaddress.ip_network(f"{s.getsockname()[0]}/24", strict=False)


def sweep(subnet: ipaddress.IPv4Network, port: int, timeout_s: float = 0.5) -> None:
    """Knock on `port` across the subnet so every live host lands in the ARP table."""

    def knock(host: ipaddress.IPv4Address) -> None:
        try:
            socket.create_connection((str(host), port), timeout=timeout_s).close()
        except OSError:
            pass

    with concurrent.futures.ThreadPoolExecutor(64) as pool:
        list(pool.map(knock, subnet.hosts()))


def resolve_stream_url(
    url: str,
    mac: str,
    lookup: Callable[[str], str | None] = arp_lookup,
    rescan: Callable[[int], None] = lambda port: sweep(local_subnet(), port),
) -> str:
    """Fill a `{host}` placeholder in the stream URL with the camera's current IP.

    The camera gets its address from DHCP, so it is found by its MAC (stable) through the
    ARP table, knocking on the stream port across the LAN first if the entry has aged out.
    Called on every (re)connect, so a new lease is picked up on the next retry. Needs the
    container on the host network so it sees the LAN and the host's ARP table.
    """
    if "{host}" not in url:
        return url
    if not mac:
        raise RuntimeError("FEEDER_STREAM_URL has {host} but FEEDER_CAMERA_MAC is not set")
    ip = lookup(mac)
    if ip is None:
        parts = urlsplit(url.replace("{host}", "placeholder"))
        rescan(parts.port or (554 if parts.scheme == "rtsp" else 443))
        ip = lookup(mac)
    if ip is None:
        raise RuntimeError(f"camera {mac} not found on the LAN")
    return url.replace("{host}", ip)


def build_ffmpeg_command(
    secrets: Secrets,
    fps: int = 2,
    width: int = FRAME_WIDTH,
    stream_url: str | None = None,
    mac: str = "",
) -> list[str]:
    """ffmpeg invocation for either an RTSP (newer Kasa) or HTTPS :19443 (older Kasa) stream.

    Defaults to the feeder camera; pass `stream_url` and `mac` for another one (the bed camera)."""
    if stream_url is None:
        stream_url, mac = secrets.feeder_stream_url, secrets.feeder_camera_mac
    url = resolve_stream_url(stream_url, mac)
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin"]
    if url.startswith("rtsp://"):
        cmd += ["-rtsp_transport", "tcp"]
    elif url.startswith("https://"):
        if secrets.kasa_email and secrets.kasa_password:
            token = base64.b64encode(
                f"{secrets.kasa_email}:{kasa_legacy_password(secrets.kasa_password)}".encode()
            ).decode("ascii")
            cmd += ["-headers", f"Authorization: Basic {token}\r\n"]
    cmd += [
        "-i",
        url,
        "-an",
        "-vf",
        f"fps={fps},scale={width}:-2",
        "-q:v",
        "5",
        "-f",
        "image2pipe",
        "-vcodec",
        "mjpeg",
        "-",
    ]
    return cmd


def split_jpegs(buffer: bytearray) -> list[bytes]:
    """Pull complete JPEGs off the front of `buffer`, leaving any partial tail in place."""
    frames: list[bytes] = []
    while True:
        start = buffer.find(JPEG_START)
        if start < 0:
            buffer.clear()
            return frames
        end = buffer.find(JPEG_END, start + 2)
        if end < 0:
            if start:
                del buffer[:start]
            return frames
        frames.append(bytes(buffer[start : end + 2]))
        del buffer[: end + 2]


class Camera(threading.Thread):
    """Runs ffmpeg forever, pushing frames into `buffer` and calling `on_frame` for each."""

    def __init__(
        self,
        secrets: Secrets,
        buffer: FrameBuffer,
        on_frame: Callable[[Frame], None] | None = None,
        clock: Callable[[], datetime] = datetime.now,
        fps: int = 2,
        stall_s: float = CAMERA_STALL_S,
        stream_url: str | None = None,
        mac: str = "",
        name: str = "camera",
    ):
        super().__init__(name=name, daemon=True)
        self.secrets = secrets
        # None = the feeder camera from `secrets`; otherwise this camera's own URL and MAC.
        self.stream_url = stream_url
        self.mac = mac
        self.buffer = buffer
        self.on_frame = on_frame
        self.clock = clock
        self.fps = max(1, fps)
        self.stall_s = max(1.0, stall_s)
        self.stop_event = threading.Event()
        self.last_error: str | None = None

    def run(self) -> None:
        backoff = 2.0
        while not self.stop_event.is_set():
            started = time.monotonic()
            try:
                self._stream_once()
            except Exception as exc:  # noqa: BLE001 - keep the camera loop alive
                self.last_error = str(exc)
                log.warning("%s stream failed: %s", self.name, exc)
            if self.stop_event.is_set():
                return
            healthy_for = time.monotonic() - started
            backoff = 2.0 if healthy_for > 60 else min(backoff * 2, 30.0)
            log.info("%s reconnecting in %.0fs", self.name, backoff)
            self.stop_event.wait(backoff)

    def _stream_once(self) -> None:
        cmd = build_ffmpeg_command(
            self.secrets, fps=self.fps, stream_url=self.stream_url, mac=self.mac
        )
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        assert proc.stdout is not None
        try:
            self._pump(proc.stdout)
        finally:
            proc.kill()
            stderr = proc.stderr.read().decode("utf-8", "replace").strip() if proc.stderr else ""
            proc.wait()
        if stderr:
            raise RuntimeError(f"ffmpeg exited: {stderr[-500:]}")
        raise RuntimeError("ffmpeg stream ended")

    def _pump(self, stdout) -> None:
        """Read JPEGs off ffmpeg's stdout, pushing frames. Raises if the stream ends OR stalls.

        A camera whose RTSP socket stays open but stops sending data (seen in the wild) would
        block a plain read() forever, so the camera never reconnects and feeding silently stops.
        select() with a tick lets us notice no bytes have arrived for `stall_s` and bail, which
        bubbles up to run()'s reconnect-with-backoff."""
        pending = bytearray()
        last_data = time.monotonic()
        tick = min(1.0, self.stall_s)
        while not self.stop_event.is_set():
            ready, _, _ = select.select([stdout], [], [], tick)
            if ready:
                chunk = stdout.read1(65536) if hasattr(stdout, "read1") else stdout.read(65536)
                if not chunk:
                    return  # EOF: ffmpeg exited; _stream_once reports why and reconnects
                last_data = time.monotonic()
                pending += chunk
                for jpeg in split_jpegs(pending):
                    frame = Frame(jpeg=jpeg, at=self.clock())
                    self.buffer.push(frame)
                    if self.on_frame:
                        self.on_frame(frame)
            elif time.monotonic() - last_data > self.stall_s:
                raise RuntimeError(f"no camera frames for {self.stall_s:.0f}s; reconnecting")

    def stop(self) -> None:
        self.stop_event.set()
