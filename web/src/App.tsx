import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import Waveform from "./Waveform";
import { ProcessDialog, RetimeDialog, UploadDialog, VideoPicker } from "./Pickers";
import { JobRow, JobsDialog, ShortcutsDialog } from "./Jobs";

type Cue = { i: number; start: number; end: number; tr: string; en: string };

type Theme = "dark" | "light";
const initialTheme = (): Theme => {
  try { const t = localStorage.getItem("theme"); if (t === "dark" || t === "light") return t; } catch { /* private window */ }
  return matchMedia("(prefers-color-scheme: light)").matches ? "light" : "dark";
};

const MAX_LINE = 42, MAX_LINES = 2, MAX_CPS = 17;

function tc(s: number) {
  const m = Math.floor(s / 60), r = s - m * 60;
  return `${String(Math.floor(m / 60)).padStart(2, "0")}:${String(m % 60).padStart(2, "0")}:${r.toFixed(1).padStart(4, "0")}`;
}

function check(c: Cue) {
  const lines = c.en.split("\n");
  const cps = c.en.replace(/\n/g, " ").length / Math.max(c.end - c.start, 0.1);
  const flags: string[] = [];
  if (lines.length > MAX_LINES || lines.some((l) => l.length > MAX_LINE)) flags.push("Too long");
  if (cps > MAX_CPS) flags.push("Fast");
  if (!c.en.trim()) flags.push("Empty");
  return { cps, flags };
}

export default function App() {
  const [episodes, setEpisodes] = useState<string[]>([]);
  const [ep, setEp] = useState("");
  const [cues, setCues] = useState<Cue[]>([]);
  const [sel, setSel] = useState(0);
  const [onlyFlagged, setOnlyFlagged] = useState(false);
  const [draft, setDraft] = useState("");
  const [status, setStatus] = useState("");
  const listRef = useRef<HTMLDivElement>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const stopAt = useRef<number | null>(null);
  const [peaks, setPeaks] = useState<{ rate: number; peaks: number[] }>({ rate: 1, peaks: [] });
  const [now, setNow] = useState(0);
  const [vfile, setVfile] = useState({ file: "", auto: true });
  const [vkey, setVkey] = useState(0); // bumps when the chosen video changes
  const [ver, setVer] = useState(0); // bumps when subtitles are re-uploaded
  const [dialog, setDialog] = useState<"" | "video" | "upload" | "retime" | "process" | "jobs" | "keys">("");
  const [jobs, setJobs] = useState<JobRow[]>([]);
  const lastStates = useRef<Record<string, string>>({});
  const [theme, setTheme] = useState<Theme>(initialTheme);
  const menu = useRef<HTMLDetailsElement>(null);

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    try { localStorage.setItem("theme", theme); } catch { /* ignore */ }
  }, [theme]);

  const loadJobs = useCallback(() => {
    fetch("/api/jobs").then((r) => r.json()).then((list: JobRow[]) => {
      const prev = lastStates.current;
      for (const j of list) { // announce a job that just finished
        if ((prev[j.id] === "running" || prev[j.id] === "queued") && j.state !== "running" && j.state !== "queued") setStatus(`Job ${j.state}: ${j.title}`);
      }
      lastStates.current = Object.fromEntries(list.map((j) => [j.id, j.state]));
      setJobs(list);
    }).catch(() => {});
  }, []);
  useEffect(() => {
    loadJobs();
    const t = setInterval(loadJobs, 4000);
    return () => clearInterval(t);
  }, [loadJobs]);
  const running = jobs.filter((j) => j.state === "running" || j.state === "queued").length;

  useEffect(() => {
    fetch("/api/episodes").then((r) => r.json()).then((e: string[]) => { setEpisodes(e); if (e[0]) setEp(e[0]); })
      .catch(() => setStatus("Cannot reach the API"));
  }, []);

  useEffect(() => {
    if (!ep) return;
    fetch(`/api/cues?ep=${encodeURIComponent(ep)}`).then((r) => r.json()).then((c: Cue[]) => { setCues(c); setSel(0); setStatus(""); })
      .catch(() => setStatus("Cannot load episode"));
  }, [ep, ver]);

  useEffect(() => {
    if (!ep) return;
    setVfile({ file: "", auto: true });
    fetch(`/api/video-choice?ep=${encodeURIComponent(ep)}`).then((r) => r.json()).then(setVfile).catch(() => setVfile({ file: "", auto: true }));
  }, [ep, vkey]);

  useEffect(() => {
    setPeaks({ rate: 1, peaks: [] });
    if (!ep) return;
    fetch(`/api/peaks?ep=${encodeURIComponent(ep)}`).then((r) => (r.ok ? r.json() : Promise.reject()))
      .then(setPeaks).catch(() => setStatus("No waveform: no video yet. Use Change video."));
  }, [ep, vkey]);

  const checks = useMemo(() => cues.map(check), [cues]);
  const visible = useMemo(() => cues.filter((c) => !onlyFlagged || checks[c.i].flags.length), [cues, checks, onlyFlagged]);
  const cue = cues[sel];
  useEffect(() => { setDraft(cue?.en ?? ""); }, [cue?.i, cue?.en]);
  useEffect(() => { listRef.current?.querySelector(`[data-i="${sel}"]`)?.scrollIntoView({ block: "nearest" }); }, [sel]);
  useEffect(() => { // selecting a cue parks the video at its start
    const v = videoRef.current;
    if (v && cue) { stopAt.current = null; v.currentTime = cue.start; }
  }, [cue?.i]); // eslint-disable-line react-hooks/exhaustive-deps

  const playCue = () => {
    const v = videoRef.current;
    if (!v || !cue) return;
    v.currentTime = cue.start; stopAt.current = cue.end; void v.play();
  };
  const onTime = () => {
    const v = videoRef.current!;
    setNow(v.currentTime);
    if (stopAt.current !== null && v.currentTime >= stopAt.current) { v.pause(); stopAt.current = null; }
  };
  const playing = cues.find((c) => c.start <= now && now < c.end);

  const move = useCallback((d: number) => {
    const at = visible.findIndex((c) => c.i === sel);
    const next = visible[Math.min(Math.max(at + d, 0), visible.length - 1)];
    if (next) setSel(next.i);
  }, [visible, sel]);

  const finishReview = async () => {
    if (!window.confirm(`Finish review of ${ep}?\n\nThis deletes its subtitles, edit backup and caches from the app's output folder. The copies in the media folder are not touched. This cannot be undone.`)) return;
    const r = await fetch(`/api/episode?ep=${encodeURIComponent(ep)}`, { method: "DELETE" });
    if (!r.ok) { setStatus(`Could not finish review: ${(await r.json()).detail ?? r.status}`); return; }
    const l: string[] = await (await fetch("/api/episodes")).json();
    setEpisodes(l); setEp(l[0] ?? ""); if (!l.length) setCues([]);
    setStatus(`Finished ${ep}`);
  };

  const save = useCallback(async (andNext: boolean) => {
    if (!cue) return;
    if (draft !== cue.en) {
      setStatus("Saving…");
      const r = await fetch(`/api/cues/${cue.i}?ep=${encodeURIComponent(ep)}`, {
        method: "PUT", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ en: draft }),
      });
      if (!r.ok) { setStatus("Save failed"); return; }
      setCues((cs) => cs.map((c) => (c.i === cue.i ? { ...c, en: draft } : c)));
      const { media_error } = await r.json();
      setStatus(media_error ? `Saved cue ${cue.i + 1}; not copied to media folder: ${media_error}` : `Saved cue ${cue.i + 1}`);
    }
    if (andNext) move(1);
  }, [cue, draft, ep, move]);

  const onKey = (e: React.KeyboardEvent) => {
    if (e.ctrlKey && e.key === "Enter") { e.preventDefault(); void save(true); }
    else if (e.ctrlKey && e.key === "ArrowDown") { e.preventDefault(); move(1); }
    else if (e.ctrlKey && e.key === "ArrowUp") { e.preventDefault(); move(-1); }
    else if (e.ctrlKey && e.key.toLowerCase() === "l") { e.preventDefault(); playCue(); }
    else if (e.key === "?" && !/^(TEXTAREA|SELECT|INPUT)$/.test((e.target as HTMLElement).tagName)) setDialog("keys");
    else if (e.key === " " && !/^(TEXTAREA|SELECT|INPUT|BUTTON)$/.test((e.target as HTMLElement).tagName)) {
      e.preventDefault();
      const v = videoRef.current;
      if (v) { stopAt.current = null; void (v.paused ? v.play() : v.pause()); }
    }
  };

  const m = cue ? check({ ...cue, en: draft }) : null;
  const flagged = checks.filter((c) => c.flags.length).length;

  return (
    <div className="app" onKeyDown={onKey}>
      <header className="top">
        <h1>Subtitle Review</h1>
        <label htmlFor="ep" className="sr">Episode</label>
        <select id="ep" value={ep} onChange={(e) => setEp(e.target.value)}>
          {episodes.map((e) => <option key={e} value={e}>{e}</option>)}
        </select>
        <button className="btn" aria-pressed={onlyFlagged} onClick={() => setOnlyFlagged((v) => !v)}>Flagged only <span className="count">{flagged}</span></button>
        <span className="status" role="status" aria-live="polite">{status || `${cues.length} cues`}</span>
        <details className="menu" ref={menu}>
          <summary className="btn primary">Tools ▾</summary>
          <div className="menulist">
            {([["upload", "Upload SRT…"], ["retime", "Re-time SRT…"], ["process", "Transcribe / re-time / translate…"]] as const).map(([k, label]) => (
              <button key={k} onClick={() => { menu.current?.removeAttribute("open"); setDialog(k); }}>{label}</button>
            ))}
          </div>
        </details>
        <button className="btn" onClick={() => setDialog("jobs")}>Jobs{running > 0 && <span className="count live" aria-label={`${running} running`}>{running}</span>}</button>
        <button className="btn icon" onClick={() => setTheme((t) => (t === "dark" ? "light" : "dark"))} aria-label={`Switch to ${theme === "dark" ? "light" : "dark"} theme`}>{theme === "dark" ? "☀" : "☾"}</button>
        <button className="btn icon" onClick={() => setDialog("keys")} aria-label="Keyboard shortcuts">?</button>
      </header>
      <div className="main">
        <div className="listwrap">
        <div className="thead" aria-hidden="true"><span>#</span><span>Time</span><span>Turkish</span><span>English</span><span /></div>
        <div className="list" ref={listRef} role="listbox" aria-label="Cues">
          {visible.length === 0 && <div className="empty">{cues.length ? "No flagged cues." : "No episode loaded."}</div>}
          {visible.map((c) => {
            const f = checks[c.i].flags;
            return (
              <button key={c.i} data-i={c.i} className="row" role="option" aria-selected={c.i === sel} onClick={() => setSel(c.i)}>
                <span className="mono num">{c.i + 1}</span>
                <span className="mono time">{tc(c.start)}</span>
                <span className="tr" lang="tr">{c.tr}</span>
                <span className="en" lang="en">{c.en}</span>
                <span className={`chip ${f.includes("Empty") ? "bad" : "warn"}`}>{f[0] ? `⚠ ${f[0]}` : ""}</span>
              </button>
            );
          })}
        </div>
        </div>
        <section className="editor" aria-label="Cue editor">
          <div className="player">
            <video ref={videoRef} src={ep && vfile.file ? `/api/video?ep=${encodeURIComponent(ep)}&v=${vkey}` : undefined} controls preload="metadata"
              onTimeUpdate={onTime} onSeeked={onTime} onError={() => vfile.file && setStatus("Video failed to load (first play converts the episode, about a minute)")} />
            {playing && <div className="overlay" lang="en">{playing.en}</div>}
          </div>
          <div className="vfile">
            <span className="hint" title={vfile.file}>{vfile.file ? `${vfile.file.split("/").pop()}${vfile.auto ? " (auto)" : ""}` : "No video found for this episode"}</span>
            <button className="btn" onClick={() => setDialog("video")} disabled={!ep}>Change video…</button>
            <button className="btn" onClick={() => void finishReview()} disabled={!ep}>Finish review…</button>
          </div>
          {cue ? (
            <>
              <div className="mono hint">#{cue.i + 1} · {tc(cue.start)} → {tc(cue.end)} · {(cue.end - cue.start).toFixed(1)} s</div>
              <label>Turkish</label>
              <div className="src" lang="tr">{cue.tr}</div>
              <label htmlFor="en">English</label>
              <textarea id="en" lang="en" value={draft} onChange={(e) => setDraft(e.target.value)} />
              <div className="meters mono">
                <span className={m!.cps > MAX_CPS ? "chip bad" : "chip"}>{m!.cps.toFixed(1)} cps (max {MAX_CPS}){m!.cps > MAX_CPS ? " — too fast" : ""}</span>
                <span className={draft.split("\n").some((l) => l.length > MAX_LINE) ? "chip bad" : "chip"}>
                  {draft.split("\n").map((l) => l.length).join(" / ")} chars (max {MAX_LINE}/line)
                </span>
              </div>
              <div className="actions">
                <button className="btn primary" onClick={() => void save(true)}>Save and next</button>
                <button className="btn" onClick={() => setDraft(cue.en)} disabled={draft === cue.en}>Revert</button>
              </div>
              <div className="hint">Press ? for keyboard shortcuts</div>
            </>
          ) : <div className="empty">Pick an episode.</div>}
        </section>
      </div>
      {dialog === "jobs" && (
        <JobsDialog jobs={jobs} onClose={() => setDialog("")} onChanged={loadJobs}
          onOpen={(e) => fetch("/api/episodes").then((r) => r.json()).then((l: string[]) => { setEpisodes(l); setEp(e); setVer((v) => v + 1); setVkey((k) => k + 1); })}
          onClear={() => void fetch("/api/jobs-clear", { method: "POST" }).then(loadJobs)} />
      )}
      {dialog === "keys" && <ShortcutsDialog onClose={() => setDialog("")} />}
      {dialog === "video" && <VideoPicker ep={ep} current={vfile.file} onClose={() => setDialog("")} onChosen={() => setVkey((k) => k + 1)} />}
      {(dialog === "upload" || dialog === "retime" || dialog === "process") && (() => {
        const done = (e: string) => {
          fetch("/api/episodes").then((r) => r.json()).then((l: string[]) => { setEpisodes(l); setEp(e); setVer((v) => v + 1); setVkey((k) => k + 1); });
        };
        const queued = () => { setStatus("Added to the queue (see Jobs)"); loadJobs(); };
        const close = () => setDialog("");
        return dialog === "upload" ? <UploadDialog existing={episodes} onClose={close} onDone={done} />
          : dialog === "retime" ? <RetimeDialog defaultVideo={vfile.file} existing={episodes} onClose={close} onQueued={queued} />
          : <ProcessDialog defaultVideo={vfile.file} onClose={close} onQueued={queued} />;
      })()}
      <Waveform video={videoRef} peaks={peaks.peaks} rate={peaks.rate} cues={cues} sel={sel} />
    </div>
  );
}
