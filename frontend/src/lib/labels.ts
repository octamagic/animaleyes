import type { Status } from "./api";
import { nameMap } from "./api";

// state -> [label, sublabel, dot tone]
export const STATES: Record<string, [string, string, string]> = {
  OUTSIDE_WINDOW: ["Off hours", "Outside the active feeding window", "muted"],
  IDLE: ["Watching", "Waiting for something to move at the bowl", "teal"],
  WATCHING: ["Checking", "Figuring out who's at the bowl", "sage"],
  OPENING: ["Opening", "Opening the feeder", "pearl"],
  VERIFYING: ["Checking plate", "Lid open — making sure the plate has food", "sage"],
  FEEDING: ["Feeding", "Lid is open — Grrr is eating", "teal"],
  CLOSING: ["Closing", "Closing the feeder", "pearl"],
  COOLDOWN: ["Cooldown", "Waiting before the next feed", "muted"],
  DONE: ["Done", "Finished for this window", "muted"],
};

export const EVENTS: Record<string, [string, string]> = {
  open: ["🍽️", "Fed"], close: ["✅", "Closed"], veto: ["🚫", "Blocked — wrong animal"],
  sighting: ["👀", "Spotted at the bowl"],
  feed_failed: ["⚠️", "Feed failed"], close_failed: ["⚠️", "Close failed"],
  rotated_empty_plate: ["🔄", "Empty plate — rotated to the next"], empty_no_food: ["🪹", "Served plate was empty"],
  wanted_food_none_left: ["🙁", "Wanted food, none left"], startup: ["▶️", "Started up"],
  bowie_during_feed: ["🐕", "Bowie during feeding"], cat_during_feed: ["🐈", "Cat during feeding"],
  camera_offline: ["📵", "Camera offline"], camera_online: ["📶", "Camera back online"],
  plates_set: ["🥣", "Plates updated"], grrr_blocked: ["⏳", "Grrr seen, not fed yet"],
  manual_open: ["🖐️", "Opened by hand"], manual_close: ["🖐️", "Closed by hand"],
  manual_close_noop: ["⚠️", "Close not sent — feeder says it's already closed"],
  manual_rotate: ["🔄", "Rotated by hand"], manual_open_failed: ["⚠️", "Manual open failed"],
  manual_close_failed: ["⚠️", "Manual close failed"], manual_rotate_failed: ["⚠️", "Manual rotate failed"],
  power_cycle: ["🔌", "Feeder power cycled"], power_cycle_failed: ["⚠️", "Power cycle failed"],
  power_cycle_skipped: ["🔌", "Power cycle skipped (too soon)"], power_on: ["🔌", "Feeder power restored"],
  power_on_failed: ["⚠️", "Could not restore feeder power"], manual_power_cycle: ["🔌", "Power cycle (dry run)"],
  config: ["⚙️", "Settings changed"],
};

export const fmtAge = (s: number | null) =>
  s == null ? "—" : s < 90 ? `${s}s ago` : `${Math.round(s / 60)}m ago`;
export const fmtIn = (s: number) => (!s ? "now" : s < 90 ? `in ${s}s` : `in ${Math.round(s / 60)}m`);

export function verdictText(s: Status): { who: string; why: string } {
  const v = s.last_verdict;
  if (!v) return { who: "Nothing yet", why: "No animal checked since the last feed" };
  // Say how old the read is, so a stale "no animal" never looks live.
  const age = s.last_verdict_age_s != null && s.last_verdict_age_s >= 10 ? ` · ${fmtAge(s.last_verdict_age_s)}` : "";
  if (v.animal === "none") return { who: "No animal at the bowl" + age, why: v.reason || "" };
  if (v.animal === "unsure") return { who: "Not sure who that is" + age, why: v.reason || "" };
  const names = nameMap(s.personas);
  const pct = Math.round((v.confidence || 0) * 100);
  const extra = v.other_animals_present?.length
    ? " · also " + v.other_animals_present.map((a) => names[a] || a).join(", ")
    : "";
  return {
    who: `${names[v.animal] || v.animal} · ${pct}% · ${v.at_bowl ? "at the bowl" : "nearby"}${extra}${age}`,
    why: v.reason || "",
  };
}

export function motionText(s: Status): string {
  if (s.motion_source !== "camera") return "frame diff";
  if (s.camera_events_ok === false) return "events down";
  if (s.camera_animal) return "🐾 animal";
  if (s.camera_motion) return "motion";
  return "quiet";
}
