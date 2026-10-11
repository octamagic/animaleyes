import { useEffect, useState } from "react";
import { api } from "../lib/api";
import { Button, Card, useToast } from "../lib/ui";

type Ctl = { key: string; label: string; type: "bool" | "number" | "text" | "time" | "select"; group: string; opts?: string[] };
const CONTROLS: Ctl[] = [
  { key: "ENABLED", label: "Enabled", type: "bool", group: "Schedule" },
  { key: "ACTIVE_START", label: "Active from", type: "time", group: "Schedule" },
  { key: "ACTIVE_END", label: "Active until", type: "time", group: "Schedule" },
  { key: "DRY_RUN", label: "Dry run (don't move feeder)", type: "bool", group: "Schedule" },
  { key: "IDENTIFIER", label: "Detector", type: "select", group: "Identification", opts: ["cascade", "yolo", "claude"] },
  { key: "GRRR_MIN_CONF", label: "Min confidence for Grrr", type: "number", group: "Identification" },
  { key: "YOLO_CLASSIFIER", label: "Local identity model (path; blank = size rule)", type: "text", group: "Identification" },
  { key: "YOLO_CLS_MIN_CONF", label: "Local identity min confidence", type: "number", group: "Identification" },
  { key: "CONFIRMATIONS_REQUIRED", label: "Confirmations before feeding", type: "number", group: "Identification" },
  { key: "CONFIRM_WINDOW_S", label: "Confirmations must fall within (sec)", type: "number", group: "Identification" },
  { key: "OPEN_REQUIRES_AT_BOWL", label: "Require head-in-bowl to open", type: "bool", group: "Identification" },
  { key: "CASCADE_PROBE_S", label: "Ask Claude when the local detector sees nothing, every (sec, 0 = never)", type: "number", group: "Identification" },
  { key: "MIN_GAP_MIN", label: "Min minutes between feeds", type: "number", group: "Feeding" },
  { key: "FEED_WITHOUT_LOADED_PLATES", label: "Feed even if no bowl is marked loaded (try each bowl once)", type: "bool", group: "Feeding" },
  { key: "LEAVE_TIMEOUT_S", label: "Close after gone (sec)", type: "number", group: "Feeding" },
  { key: "FEEDING_MAX_MIN", label: "Max feeding length (min)", type: "number", group: "Feeding" },
  { key: "FEEDING_POLL_S", label: "Check interval while feeding (sec)", type: "number", group: "Feeding" },
  { key: "VERIFY_FOOD", label: "Verify a plate has food after opening", type: "bool", group: "Plate verification" },
  { key: "VERIFY_EMPTY_CONFIRMATIONS", label: "Empty reads before rotating", type: "number", group: "Plate verification" },
  { key: "MAX_ROTATE_FOR_FOOD", label: "Max rotations hunting for food", type: "number", group: "Plate verification" },
  { key: "VERIFY_POLL_S", label: "Food check interval (sec)", type: "number", group: "Plate verification" },
  { key: "VERIFY_TIMEOUT_S", label: "Give up verifying after (sec)", type: "number", group: "Plate verification" },
  { key: "CAMERA_FPS", label: "Live video FPS (restart to apply)", type: "number", group: "Motion & sensing" },
  { key: "MOTION_SOURCE", label: "Motion source", type: "select", group: "Motion & sensing", opts: ["camera", "frames"] },
  { key: "MOTION_HOLD_S", label: "Keep watching after motion (sec)", type: "number", group: "Motion & sensing" },
  { key: "LLM_MIN_INTERVAL_S", label: "Min seconds between checks", type: "number", group: "Motion & sensing" },
  { key: "COLLECT_TRAINING", label: "Auto-collect training frames", type: "bool", group: "Training" },
  { key: "CAPTURE_MODE", label: "Capture mode (save animal frames to tag)", type: "bool", group: "Training" },
  { key: "CAPTURE_MIN_GAP_S", label: "Min seconds between captures", type: "number", group: "Training" },
  { key: "ALWAYS_WATCH", label: "Always watch & log animals (local model, incl. off hours)", type: "bool", group: "Training" },
  { key: "WATCH_INTERVAL_S", label: "Seconds between always-on checks", type: "number", group: "Training" },
  { key: "LID_POLL_S", label: "Lid check interval (sec)", type: "number", group: "Advanced" },
  { key: "FEED_RETRY_BACKOFF_S", label: "Backoff after a failed feed (sec)", type: "number", group: "Advanced" },
  { key: "POWER_CYCLE_ON_FAULT", label: "Power-cycle the feeder on a device fault", type: "bool", group: "Feeder power" },
  { key: "POWER_CYCLE_OFF_S", label: "Keep it off for (sec)", type: "number", group: "Feeder power" },
  { key: "POWER_CYCLE_BOOT_S", label: "Wait for it to boot before commands (sec)", type: "number", group: "Feeder power" },
  { key: "POWER_CYCLE_MIN_GAP_MIN", label: "At most one automatic cycle per (min)", type: "number", group: "Feeder power" },
  { key: "HEARTBEAT_MIN", label: "Heartbeat interval (min)", type: "number", group: "Advanced" },
  { key: "DASH_AUTH", label: "Dashboard auth", type: "select", group: "Advanced", opts: ["tailscale", "basic", "none"] },
];

export function Settings() {
  const toast = useToast();
  const [open, setOpen] = useState(false);
  const [cfg, setCfg] = useState<Record<string, unknown>>({});
  const load = () => api.config().then(setCfg);
  useEffect(() => { load(); }, []);

  const set = (k: string, v: unknown) => setCfg((c) => ({ ...c, [k]: v }));
  async function save() {
    const body: Record<string, unknown> = {};
    for (const c of CONTROLS) if (c.key in cfg) body[c.key] = cfg[c.key];
    const r = await api.post("/api/config", body);
    if (!r.ok) toast("Rejected: " + (await r.text()), true);
    else { toast("Settings saved"); load(); }
  }

  let group = "";
  return (
    <Card>
      <button className="h2 flex items-center gap-2 w-full text-left" onClick={() => setOpen((x) => !x)}>
        Settings <span className="text-muted">{open ? "▴" : "▾"}</span>
      </button>
      {open && (
        <>
          {CONTROLS.map((c) => {
            if (!(c.key in cfg)) return null;
            const head = c.group !== group ? ((group = c.group), c.group) : null;
            return (
              <div key={c.key}>
                {head && <div className="text-teal text-[11px] font-bold uppercase tracking-wider mt-3 mb-1">{head}</div>}
                <div className="flex justify-between items-center gap-3 py-2 border-b border-edge">
                  <label className="text-sm">{c.label}</label>
                  {c.type === "bool" ? (
                    <input type="checkbox" className="w-5 h-5" checked={!!cfg[c.key]} onChange={(e) => set(c.key, e.target.checked)} />
                  ) : c.type === "select" ? (
                    <select className="input" value={String(cfg[c.key])} onChange={(e) => set(c.key, e.target.value)}>
                      {c.opts!.map((o) => <option key={o}>{o}</option>)}
                    </select>
                  ) : c.type === "time" ? (
                    <input className="input" type="time" value={String(cfg[c.key] ?? "")} onChange={(e) => set(c.key, e.target.value)} />
                  ) : (
                    <input className="input" type={c.type} step="any" value={String(cfg[c.key])} onChange={(e) => set(c.key, e.target.value)} />
                  )}
                </div>
              </div>
            );
          })}
          <div className="mt-3"><Button variant="primary" sm onClick={save}>Save settings</Button></div>
        </>
      )}
    </Card>
  );
}
