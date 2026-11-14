import { useEffect, useRef, useState } from "react";
import type { Cam, Status } from "../lib/api";
import { api } from "../lib/api";
import { Card, useToast } from "../lib/ui";
import { STATES, bedText, verdictText } from "../lib/labels";

const DOT: Record<string, string> = {
  teal: "bg-teal", sage: "bg-sage", pearl: "bg-pearl", muted: "bg-muted", bad: "bg-bad",
};

export function StateBanner({ s }: { s: Status | null }) {
  const [label, sub, tone] = (s && STATES[s.state]) || ["…", "", "muted"];
  return (
    <div className="card bg-gradient-to-br from-teal/10 to-pearl/5">
      <div className="flex items-center gap-3">
        <span className={`w-3 h-3 rounded-full shrink-0 ${DOT[tone]} shadow-[0_0_0_4px_rgba(255,255,255,0.04)]`} />
        <div>
          <div className="text-[22px] font-extrabold leading-none">{label}</div>
          <div className="text-muted text-[13px] mt-1">{sub}</div>
        </div>
      </div>
    </div>
  );
}

const CAM_NAMES: Record<Cam, string> = { feeder: "Feeder", bed: "Grrr's bed" };

/** One button per camera under the live view: a still that refreshes every few seconds, its
 *  name, and what that camera currently knows. Tapping one puts it in the big view. */
function CamPicker({ s, cam, setCam }: { s: Status; cam: Cam; setCam: (c: Cam) => void }) {
  const [tick, setTick] = useState(0);
  useEffect(() => {
    const id = setInterval(() => { if (!document.hidden) setTick((t) => t + 1); }, 4000);
    return () => clearInterval(id);
  }, []);
  const line: Record<Cam, string> = { feeder: verdictText(s).who, bed: bedText(s).who };
  return (
    <div className="grid grid-cols-2 gap-2 mt-2.5">
      {(["feeder", "bed"] as Cam[]).map((c) => (
        <button
          key={c}
          onClick={() => setCam(c)}
          aria-pressed={cam === c}
          className={`flex items-center gap-2.5 p-1.5 rounded-xl border text-left min-w-0 ${
            cam === c ? "border-teal bg-teal/10" : "border-edge"
          }`}
        >
          <img
            src={`/frame.jpg?cam=${c}&t=${tick}`}
            alt=""
            className="w-16 aspect-video object-cover rounded-lg bg-black shrink-0"
          />
          <span className="min-w-0">
            <span className="block text-sm font-bold">{CAM_NAMES[c]}</span>
            <span className="block text-muted text-xs truncate">{line[c]}</span>
          </span>
        </button>
      ))}
    </div>
  );
}

export function LiveView({ s, cam, setCam }: { s: Status | null; cam: Cam; setCam: (c: Cam) => void }) {
  const img = useRef<HTMLImageElement>(null);
  const box = useRef<HTMLDivElement>(null);
  const [live, setLive] = useState(true);

  const start = () => { if (img.current) img.current.src = `/stream.mjpg?cam=${cam}&t=` + Date.now(); };
  // iPhone Safari has no element fullscreen (only <video>), so the button used to do nothing
  // there. Fall back to taking over the viewport with CSS wherever the real API is missing
  // or refuses.
  const [expanded, setExpanded] = useState(false);
  const fullscreen = () => {
    const el = box.current;
    if (!el) return;
    if (document.fullscreenElement) { document.exitFullscreen(); return; }
    if (expanded) { setExpanded(false); return; }
    if (el.requestFullscreen) el.requestFullscreen().catch(() => setExpanded(true));
    else setExpanded(true);
  };
  useEffect(() => {
    if (!expanded) return;
    const prev = document.body.style.overflow;
    document.body.style.overflow = "hidden"; // no page scroll behind the takeover
    return () => { document.body.style.overflow = prev; };
  }, [expanded]);
  useEffect(() => {
    if (live) start();
    const onVis = () => { if (live && !document.hidden) start(); };
    document.addEventListener("visibilitychange", onVis);
    return () => document.removeEventListener("visibilitychange", onVis);
  }, [live, cam]);
  // while paused, refresh a still alongside status
  useEffect(() => {
    if (live || !img.current) return;
    img.current.src = `/frame.jpg?cam=${cam}&t=` + Date.now();
  }, [s, live, cam]);

  const v = !s ? { who: "Waiting…", why: "" } : cam === "bed" ? bedText(s) : verdictText(s);
  const toast = useToast();
  // The bed check compares against a photo of the empty beds; this saves the current frame as it.
  async function saveEmptyBeds() {
    if (!confirm("Are both beds empty right now? This saves the current picture as the “empty beds” photo the bed check compares against.")) return;
    const r = await api.post("/api/bed/reference");
    toast(r.ok ? "Saved the empty-beds photo" : "Couldn't save it", !r.ok);
  }
  return (
    <Card>
      <div
        ref={box}
        className={
          expanded
            ? "live-box fixed inset-0 z-[55] overflow-hidden bg-black"
            : "live-box relative rounded-xl overflow-hidden bg-black aspect-video"
        }
      >
        <img
          ref={img}
          alt="live camera"
          className="w-full h-full object-contain block"
          onError={() => live && setTimeout(start, 3000)}
        />
        {live && (
          <span className="absolute top-2.5 left-2.5 text-xs font-bold text-teal bg-black/70 backdrop-blur px-2.5 py-1 rounded-full">
            ● LIVE{s?.bed ? ` · ${CAM_NAMES[cam]}` : ""}
          </span>
        )}
        <div className="absolute top-2.5 right-2.5 flex gap-1.5">
          <button onClick={() => setLive((x) => !x)} className="btn btn-sm bg-black/60 backdrop-blur">
            {live ? "Pause" : "Resume"}
          </button>
          <button onClick={fullscreen} className="btn btn-sm bg-black/60 backdrop-blur" title="fullscreen">
            {expanded ? "✕" : "⛶"}
          </button>
        </div>
        <div className="absolute left-2.5 right-2.5 bottom-2.5 bg-black/70 backdrop-blur px-3 py-2 rounded-[10px]">
          <div className="font-bold text-sm">{v.who}</div>
          {v.why && <div className="text-muted text-xs mt-0.5">{v.why}</div>}
        </div>
      </div>
      {s?.bed && <CamPicker s={s} cam={cam} setCam={setCam} />}
      {s?.bed && cam === "bed" && (
        <div className="flex items-center gap-2.5 mt-2.5">
          <button onClick={saveEmptyBeds} className="btn btn-sm shrink-0">Beds are empty now</button>
          <span className="text-muted text-xs">
            {s.bed.has_reference
              ? "An empty-beds photo is saved. Save a new one if the bedding or camera moves."
              : "No empty-beds photo yet: save one while she's up so the check can tell empty from hidden."}
          </span>
        </div>
      )}
    </Card>
  );
}
