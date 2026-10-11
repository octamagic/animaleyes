"""Power-cycling the feeder through a smart plug.

The PetLibro's motor and radio wedge now and then (motor error 2043 "cannot close door", device
response timeouts), and the only cure is pulling the plug. A smart plug on the feeder's power
lets the state machine do that itself, the moment such a device-side error comes back, instead
of someone getting up at 2 am. Only the machine calls this (same invariant as the feeder).

Two drivers: a TP-Link Kasa/Tapo plug via python-kasa, or a pair of shell commands for any
other ecosystem (Home Assistant, Shelly, a curl to a webhook...).
"""

from __future__ import annotations

import asyncio
import logging
import subprocess
from typing import Protocol

log = logging.getLogger(__name__)


class PowerError(Exception):
    pass


class PowerSwitch(Protocol):
    def turn_off(self) -> None: ...
    def turn_on(self) -> None: ...


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
    host: str, username: str = "", password: str = "", off_cmd: str = "", on_cmd: str = ""
) -> PowerSwitch | None:
    """The configured plug, or None when no plug is set up (then faults are only reported)."""
    if off_cmd and on_cmd:
        log.info("feeder power: shell-command plug")
        return CommandPlug(off_cmd, on_cmd)
    if host:
        log.info("feeder power: Kasa/Tapo plug at %s", host)
        return KasaPlug(host, username, password)
    return None
