import type { Status } from "../lib/api";
import { Card, Tile } from "../lib/ui";
import { bedText, fmtAge, fmtIn, motionText } from "../lib/labels";

const DETECTOR: Record<string, string> = {
  cascade: "Cascade (local + Claude)", yolo: "YOLO (local)", claude: "Claude",
};

export function StatusTiles({ s }: { s: Status | null }) {
  const tiles: [string, string][] = s
    ? [
        ["Window", s.active_window + (s.in_window ? " · active" : " · off")],
        ["Next feed", s.next_allowed_feed_in_s ? fmtIn(s.next_allowed_feed_in_s) : "now"],
        ["Loaded plates", s.loaded_plates.length ? s.loaded_plates.join(", ") : "none"],
        ["Camera", s.camera_age_s == null ? "no frames" : fmtAge(s.camera_age_s)],
        ...(s.bed ? [["Grrr's bed", bedText(s).who.replace(/^Grrr is /, "")] as [string, string]] : []),
        ["Motion", motionText(s)],
        ["Detector", DETECTOR[s.identifier] || s.identifier || "—"],
        ["Checks today", `${s.llm_calls_today}${s.llm_cost_today_usd ? " · $" + s.llm_cost_today_usd.toFixed(2) : ""}`],
        ["Heartbeat", fmtAge(s.heartbeat_age_s)],
      ]
    : [];
  return (
    <Card title="Status">
      <div className="grid gap-2.5 [grid-template-columns:repeat(auto-fit,minmax(130px,1fr))]">
        {tiles.map(([l, v], i) => <Tile key={l} label={l} value={v} i={i} />)}
      </div>
    </Card>
  );
}
