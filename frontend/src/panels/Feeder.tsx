import { useEffect, useState } from "react";
import type { Status } from "../lib/api";
import { api } from "../lib/api";
import { Button, Card } from "../lib/ui";
import { useToast } from "../lib/ui";

export function Feeder({ s, refresh }: { s: Status | null; refresh: () => void }) {
  const toast = useToast();
  const pokeRefresh = () => [700, 1400, 2100, 2800, 3500, 4200].forEach((d) => setTimeout(refresh, d));

  async function act(action: string) {
    const verb = action === "power_cycle" ? "Power-cycle" : action[0].toUpperCase() + action.slice(1);
    if (!confirm(`${verb} the feeder now?`)) return;
    await api.post(`/api/feeder/${action}`);
    toast(`${action === "power_cycle" ? "Power cycle" : action[0].toUpperCase() + action.slice(1)} requested`);
    pokeRefresh();
  }
  async function resetCycle() {
    if (
      !confirm(
        "Reset tonight's feeding cycle? Closes the lid if it is open, puts the bowls served tonight back to loaded, and clears the feed count and the minimum-gap timer. Use this when Grrr was fed before you were actually in bed.",
      )
    )
      return;
    await api.post("/api/reset-cycle");
    toast("Cycle reset requested");
    pokeRefresh();
  }
  async function inBed() {
    const wait = s?.feed_start === "bedtime" ? ` The first feed waits until Grrr has been unseen for ${s?.settle?.settle_min ?? 60} min.` : "";
    if (!confirm(`We're in bed? This also resets tonight's cycle (lid closed, bowls back to loaded, feed count and minimum gap cleared).${wait}`)) return;
    await api.post("/api/bedtime");
    toast("Good night");
    pokeRefresh();
  }
  async function feedNow() {
    if (!confirm("Feed now? This opens the feeder, skipping identification.")) return;
    await api.post("/api/feed-now");
    toast("Feed requested");
    pokeRefresh();
  }

  const feeds = s?.feeds_this_window ?? 0;
  const settle = s?.settle;
  return (
    <Card title="Feeder">
      <div className="flex flex-wrap gap-2 mb-3">
        <span className={`pill ${s?.lid_open ? "text-teal border-teal/40" : "text-muted"}`}>
          {s?.lid_open ? "lid open" : "lid closed"}
        </span>
        <span className="pill text-sage">{s?.current_plate ? `plate ${s.current_plate}` : "plate —"}</span>
        <span className="pill text-beige">{feeds} feed{feeds === 1 ? "" : "s"} tonight</span>
        <span className={`pill ${s?.dry_run ? "text-pearl border-pearl/40" : "text-teal border-teal/40"}`}>
          {s?.dry_run ? "dry run" : "live"}
        </span>
        {s?.plug_off && <span className="pill text-bad border-bad/40">power cycling</span>}
        {!s?.plug_off && s?.feeder_booting && (
          <span className="pill text-pearl border-pearl/40">feeder rebooting</span>
        )}
        {settle && !settle.bedtime_at && <span className="pill text-pearl border-pearl/40">waiting for bed</span>}
        {settle?.settling && (
          <span className="pill text-pearl border-pearl/40">
            settling {Math.floor((settle.quiet_s ?? 0) / 60)}/{settle.settle_min} min
            {settle.grrr_seen_at && ` · seen ${settle.grrr_seen_at.slice(11, 16)}`}
          </span>
        )}
        {settle?.settled_at && <span className="pill text-sage">settled {settle.settled_at.slice(11, 16)}</span>}
        {s?.feed_retry && (
          <span className="pill text-pearl border-pearl/40">
            feed retry pending ({s.feed_retry.attempts} failed)
          </span>
        )}
      </div>
      <div className="flex flex-wrap gap-2">
        <Button variant="primary" onClick={inBed}>We're in bed</Button>
        <Button variant="primary" onClick={feedNow}>Feed now</Button>
        <Button onClick={() => act("open")}>Open</Button>
        <Button onClick={() => act("close")}>Close</Button>
        <Button onClick={() => act("rotate")}>Rotate</Button>
        {s?.power_cycle_available && <Button onClick={() => act("power_cycle")}>Power cycle</Button>}
        <Button variant="danger" onClick={resetCycle}>Reset cycle</Button>
      </div>
      <div className="text-muted text-xs mt-2.5">
        Buttons ask the feeder to act on its next check (a second or two). Nothing moves in dry‑run mode.
        Reset cycle starts the night over: lid closed, tonight's bowls back to loaded, feed count and
        minimum gap cleared. Untick a bowl afterwards if she really did finish it. We're in bed does the
        same reset and, in bedtime mode, holds the first feed until Grrr has gone to sleep and come back.
      </div>
    </Card>
  );
}

export function Plates({ s, refresh }: { s: Status | null; refresh: () => void }) {
  const toast = useToast();
  const [plates, setPlates] = useState<Record<string, string>>({});

  // seed from status once
  useEffect(() => {
    if (s && Object.keys(plates).length === 0) setPlates(s.plates);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [s]);

  const toggle = (p: number) =>
    setPlates((x) => ({ ...x, [p]: x[String(p)] === "loaded" ? "empty" : "loaded" }));
  async function save() {
    const loaded = [1, 2, 3].filter((p) => plates[String(p)] === "loaded");
    await api.post("/api/plates", { loaded });
    setPlates({});
    toast("Loaded plates saved");
    refresh();
  }

  return (
    <Card title="Food loaded">
      <div className="flex flex-wrap gap-2">
        {[1, 2, 3].map((p) => {
          const on = plates[String(p)] === "loaded";
          return (
            <button
              key={p}
              onClick={() => toggle(p)}
              className={`px-3.5 min-h-[42px] rounded-[10px] border font-bold flex items-center gap-2
                ${on ? "bg-sage/15 text-sage border-sage/40" : "bg-surface2 text-ink border-edge"}`}
            >
              Bowl {p}<span className="opacity-70 text-xs">{on ? "full" : "empty"}</span>
            </button>
          );
        })}
      </div>
      <div className="mt-3"><Button variant="primary" sm onClick={save}>Save loaded plates</Button></div>
      <div className="text-muted text-xs mt-2.5">
        Mark which bowls you filled. Saving resets the night's count and re‑arms the feeder.
        If you forget, the feeder still tries: when Grrr shows up it opens the bowl under the lid,
        checks it for food, and moves to the next bowl if that one is empty.
        {s?.plates_tried?.length ? <> Tried this window: {s.plates_tried.join(", ")}.</> : null}
      </div>
    </Card>
  );
}
