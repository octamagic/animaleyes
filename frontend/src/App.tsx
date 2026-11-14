import { useEffect, useState } from "react";
import { api, usePoll } from "./lib/api";
import type { Cam, Status } from "./lib/api";
import type { ThemeDef } from "./lib/theme";
import { Badge, ThemeToggle } from "./lib/ui";
import { Themes } from "./panels/Themes";
import { LiveView, StateBanner } from "./panels/LiveState";
import { CameraPtz } from "./panels/CameraPtz";
import { Feeder, Plates } from "./panels/Feeder";
import { StatusTiles } from "./panels/Status";
import { Activity } from "./panels/Activity";
import { ReferencePhotos, TrainingGallery } from "./panels/Photos";
import { Personas } from "./panels/Personas";
import { Settings } from "./panels/Settings";

export function App() {
  const [s, refresh] = usePoll<Status>(() => api.status(), 1500);
  const [themes, setThemes] = useState<ThemeDef[]>([]);
  const reloadThemes = () =>
    fetch("/api/themes").then((r) => r.json()).then(setThemes).catch(() => {});
  useEffect(() => { reloadThemes(); }, []);
  // Which camera the big live view (and the pan/tilt pad) is on.
  const [cam, setCam] = useState<Cam>("feeder");
  const shown: Cam = s?.bed ? cam : "feeder";

  return (
    <div className="max-w-[1080px] mx-auto p-4">
      <header className="flex items-center gap-3 flex-wrap mb-4">
        <h1 className="text-[26px] font-black tracking-tight flex items-center gap-2">
          <span>🐾</span> <span className="brand">animaleyes</span> <span>👁️</span>
        </h1>
        <span className="flex-1" />
        {s && <Badge tone={s.dry_run ? "pearl" : "teal"}>{s.dry_run ? "Dry run" : "Live"}</Badge>}
        {s && <Badge tone={s.camera_offline ? "bad" : "teal"}>{s.camera_offline ? "Camera offline" : "Camera ok"}</Badge>}
        {s?.bed?.camera_offline && <Badge tone="bad">Bed camera offline</Badge>}
        {s?.capture_mode && <Badge tone="sage">📸 Capturing</Badge>}
        <ThemeToggle custom={themes} />
      </header>

      <div className="grid gap-4 md:[grid-template-columns:1.25fr_1fr] md:items-start">
        <div className="grid gap-4 min-w-0">
          <StateBanner s={s} />
          <LiveView s={s} cam={shown} setCam={setCam} />
          <CameraPtz s={s} cam={shown} />
          <Feeder s={s} refresh={refresh} />
          <Plates s={s} refresh={refresh} />
        </div>
        <div className="grid gap-4 min-w-0">
          <StatusTiles s={s} />
          <Activity />
        </div>
      </div>

      <div className="grid gap-4 mt-4">
        <Personas personas={s?.personas || []} refresh={refresh} />
        <ReferencePhotos personas={s?.personas || []} counts={s?.reference_counts || {}} refresh={refresh} />
        <TrainingGallery personas={s?.personas || []} counts={s?.training_counts || {}} captureMode={s?.capture_mode} />
        <Themes themes={themes} refresh={reloadThemes} />
        <Settings />
      </div>
    </div>
  );
}
