"""Power-cycling the feeder through a smart plug.

The PetLibro's motor and radio wedge now and then (motor error 2043 "cannot close door", device
response timeouts, replies that never come), and the only cure is pulling the plug. A smart plug
on the feeder's power lets the state machine do that itself, the moment a command fails, instead
of someone getting up at 2 am. Only the machine calls this (same invariant as the feeder).

Drivers: a Shelly plug over its local HTTP API (the one on the feeder; found by MAC so a new
DHCP lease can't lose it), a TP-Link Kasa/Tapo plug via python-kasa, or a pair of shell commands
for any other ecosystem (Home Assistant, a curl to a webhook...).
"""

from __future__ import annotations

import asyncio
import json
import logging
import socket
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Protocol

log = logging.getLogger(__name__)

ARP_TABLE = "/proc/net/arp"


class PowerError(Exception):
    pass


class PowerSwitch(Protocol):
    def turn_off(self) -> None: ...
    def turn_on(self) -> None: ...


def _norm_mac(mac: str) -> str:
    return "".join(c for c in mac.lower() if c in "0123456789abcdef")


class ShellyPlug:
    """A Shelly plug (Gen1 or Gen2+: Plus, Pro, Gen3, Gen4) on the LAN, driven over its local HTTP
    API: no cloud, no account. Identified by MAC: the plug's own `/shelly` reply carries it, so
    every command goes to a host that has proven it is the right device. `host` is only the first
    place to look; when it stops answering, the ARP table (nudged with a subnet sweep) finds the
    plug's current address."""

    def __init__(
        self,
        mac: str = "",
        host: str = "",
        timeout_s: float = 5.0,
        arp_table: str = ARP_TABLE,
        sweep: bool = True,
        port: int = 80,
    ):
        if not mac and not host:
            raise PowerError("Shelly plug needs SMART_PLUG_MAC or SMART_PLUG_HOST")
        self.mac = _norm_mac(mac)
        self.host = host  # last known good address (verified before use)
        self.timeout_s = timeout_s
        self.arp_table = arp_table
        self.sweep = sweep
        self.port = port  # for tests; a real plug listens on 80
        self.gen = 2

    # http ------------------------------------------------------------------
    def _get(self, host: str, path: str) -> dict[str, Any]:
        authority = host if ":" in host or self.port == 80 else f"{host}:{self.port}"
        url = f"http://{authority}{path}"
        try:
            with urllib.request.urlopen(url, timeout=self.timeout_s) as resp:  # noqa: S310 - LAN
                return json.loads(resp.read().decode() or "{}")
        except (urllib.error.URLError, OSError, ValueError) as exc:
            raise PowerError(f"{url}: {exc}") from exc

    def _is_our_plug(self, host: str) -> bool:
        try:
            info = self._get(host, "/shelly")
        except PowerError:
            return False
        mac = _norm_mac(str(info.get("mac", "")))
        if not mac:
            return False
        if self.mac and mac != self.mac:
            return False
        self.gen = int(info.get("gen") or 1)
        self.mac = mac
        return True

    # finding it ---------------------------------------------------------------
    def _arp_hosts(self) -> list[str]:
        """Addresses the kernel currently maps to our MAC."""
        if not self.mac:
            return []
        hosts = []
        try:
            with open(self.arp_table) as f:
                next(f, None)  # header
                for line in f:
                    parts = line.split()
                    if len(parts) >= 4 and _norm_mac(parts[3]) == self.mac:
                        hosts.append(parts[0])
        except OSError as exc:
            log.debug("arp table unreadable: %s", exc)
        return hosts

    def _nudge_subnets(self) -> None:
        """Make the kernel ARP every address on the LAN by sending an empty UDP packet to each one.
        Needs no raw sockets or ping binary; ~1 s later the ARP table is fresh."""
        prefixes: set[str] = set()
        try:
            with open(self.arp_table) as f:
                next(f, None)
                for line in f:
                    ip = line.split()[0]
                    if ip.count(".") == 3 and not ip.startswith(("172.1", "172.2", "172.3")):
                        prefixes.add(ip.rsplit(".", 1)[0])
        except OSError:
            pass
        if self.host and self.host.count(".") == 3:
            prefixes.add(self.host.rsplit(".", 1)[0])
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            for prefix in sorted(prefixes)[:4]:
                for i in range(1, 255):
                    try:
                        sock.sendto(b"", (f"{prefix}.{i}", 9))
                    except OSError:
                        pass
        finally:
            sock.close()
        time.sleep(1.5)

    def resolve(self) -> str:
        """The plug's current address, verified by asking it for its MAC."""
        tried: list[str] = []
        candidates = [self.host] if self.host else []
        candidates += [h for h in self._arp_hosts() if h not in candidates]
        for host in candidates:
            tried.append(host)
            if self._is_our_plug(host):
                self.host = host
                return host
        if self.mac and self.sweep:
            self._nudge_subnets()
            for host in self._arp_hosts():
                if host not in tried and self._is_our_plug(host):
                    tried.append(host)
                    log.info("Shelly plug %s moved to %s", self.mac, host)
                    self.host = host
                    return host
        raise PowerError(
            f"Shelly plug {self.mac or self.host} not found on the LAN (tried {tried or 'nothing'})"
        )

    # switching ----------------------------------------------------------------
    def _switch(self, host: str, on: bool) -> bool:
        """Send the command and return the plug's reported output state afterwards."""
        if self.gen >= 2:
            self._get(host, f"/rpc/Switch.Set?id=0&on={'true' if on else 'false'}")
            status = self._get(host, "/rpc/Switch.GetStatus?id=0")
            return bool(status.get("output"))
        status = self._get(host, f"/relay/0?turn={'on' if on else 'off'}")
        return bool(status.get("ison"))

    def _set(self, on: bool) -> None:
        host = self.resolve()
        try:
            state = self._switch(host, on)
        except PowerError:
            # The address may have changed between the check and the command; look once more.
            self.host = ""
            host = self.resolve()
            state = self._switch(host, on)
        if state != on:
            want, got = ("on" if on else "off"), ("on" if state else "off")
            raise PowerError(f"Shelly plug at {host} still reports {got} after turn_{want}")

    def turn_off(self) -> None:
        self._set(False)

    def turn_on(self) -> None:
        self._set(True)

    def status(self) -> dict[str, Any]:
        """Output state and power draw (Gen2+: apower in W), for diagnostics."""
        host = self.resolve()
        if self.gen >= 2:
            return self._get(host, "/rpc/Switch.GetStatus?id=0")
        return self._get(host, "/relay/0")


class KasaPlug:
    """A TP-Link Kasa (HS/KP/EP series) or Tapo (P1xx, needs credentials) plug on the LAN."""

    def __init__(self, host: str, username: str = "", password: str = "", timeout_s: float = 15):
        self.host = host
        self.username = username
        self.password = password
        self.timeout_s = timeout_s

    async def _set(self, on: bool) -> None:
        from kasa import Credentials, Discover  # lazy: optional dependency

        creds = Credentials(self.username, self.password) if self.username else None
        dev = await Discover.discover_single(self.host, credentials=creds, timeout=self.timeout_s)
        try:
            await (dev.turn_on() if on else dev.turn_off())
            await dev.update()
            if dev.is_on != on:
                raise PowerError(f"plug {self.host} did not turn {'on' if on else 'off'}")
        finally:
            await dev.disconnect()

    def _run(self, on: bool) -> None:
        try:
            asyncio.run(asyncio.wait_for(self._set(on), timeout=self.timeout_s + 5))
        except PowerError:
            raise
        except Exception as exc:  # noqa: BLE001 - whatever the library raises, report and move on
            raise PowerError(f"plug {self.host}: {exc}") from exc

    def turn_off(self) -> None:
        self._run(False)

    def turn_on(self) -> None:
        self._run(True)


class CommandPlug:
    """Any plug you can switch from a shell command (`SMART_PLUG_OFF_CMD` / `SMART_PLUG_ON_CMD`)."""

    def __init__(self, off_cmd: str, on_cmd: str, timeout_s: float = 30):
        self.off_cmd = off_cmd
        self.on_cmd = on_cmd
        self.timeout_s = timeout_s

    def _run(self, cmd: str) -> None:
        try:
            proc = subprocess.run(
                ["sh", "-c", cmd], capture_output=True, text=True, timeout=self.timeout_s
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise PowerError(f"{cmd}: {exc}") from exc
        if proc.returncode != 0:
            raise PowerError(f"{cmd} failed: {proc.stderr.strip()[-300:]}")

    def turn_off(self) -> None:
        self._run(self.off_cmd)

    def turn_on(self) -> None:
        self._run(self.on_cmd)


def build_power_switch(
    kind: str = "shelly",
    host: str = "",
    mac: str = "",
    username: str = "",
    password: str = "",
    off_cmd: str = "",
    on_cmd: str = "",
) -> PowerSwitch | None:
    """The configured plug, or None when no plug is set up (then faults are only reported)."""
    if off_cmd and on_cmd:
        log.info("feeder power: shell-command plug")
        return CommandPlug(off_cmd, on_cmd)
    kind = (kind or "shelly").lower()
    if kind == "kasa":
        if not host:
            return None
        log.info("feeder power: Kasa/Tapo plug at %s", host)
        return KasaPlug(host, username, password)
    if kind == "shelly":
        if not (mac or host):
            return None
        log.info("feeder power: Shelly plug %s", mac or host)
        return ShellyPlug(mac, host)
    raise PowerError(f"unknown SMART_PLUG_KIND {kind!r} (shelly, kasa)")
