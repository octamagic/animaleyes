"""Save a frame from the camera every N seconds so you can pick reference photos.

Usage: python tools/capture.py [--camera bed] [--every 10] [--minutes 60] [--out data/captures]
Reads the camera settings from .env. Prints each saved path.
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from animaleyes.camera import Camera, FrameBuffer  # noqa: E402
from animaleyes.config import Secrets  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--every", type=float, default=10, help="seconds between saves")
    parser.add_argument("--minutes", type=float, default=60, help="how long to run")
    parser.add_argument("--out", type=Path, default=Path("data/captures"))
    parser.add_argument("--once", action="store_true", help="save one frame and exit")
    parser.add_argument("--camera", choices=("feeder", "bed"), default="feeder")
    args = parser.parse_args()

    load_dotenv()
    secrets = Secrets.from_env()
    bed = args.camera == "bed"
    if not (secrets.bed_stream_url if bed else secrets.kasa_stream_url):
        raise SystemExit(f"{'BED' if bed else 'KASA'}_STREAM_URL is not set in .env")
    args.out.mkdir(parents=True, exist_ok=True)

    buffer = FrameBuffer()
    if bed:
        camera = Camera(
            secrets, buffer, stream_url=secrets.bed_stream_url, mac=secrets.bed_camera_mac
        )
    else:
        camera = Camera(secrets, buffer)
    camera.start()
    deadline = time.monotonic() + args.minutes * 60
    last_saved: datetime | None = None
    try:
        while time.monotonic() < deadline:
            frames = buffer.latest(1)
            if frames and frames[0].at != last_saved:
                frame = frames[0]
                path = args.out / f"{frame.at.strftime('%Y%m%d-%H%M%S')}.jpg"
                path.write_bytes(frame.jpeg)
                last_saved = frame.at
                print(path, f"{len(frame.jpeg)} bytes", flush=True)
                if args.once:
                    return
                time.sleep(args.every)
            else:
                if camera.last_error:
                    print("camera:", camera.last_error, file=sys.stderr, flush=True)
                    camera.last_error = None
                time.sleep(0.5)
    finally:
        camera.stop()


if __name__ == "__main__":
    main()
