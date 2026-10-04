import { useCallback, useEffect, useMemo, useRef, useState } from "react";

type Cue = { i: number; start: number; end: number; tr: string; en: string };

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

  useEffect(() => {
    fetch("/api/episodes").then((r) => r.json()).then((e: string[]) => { setEpisodes(e); if (e[0]) setEp(e[0]); })
      .catch(() => setStatus("Cannot reach the API"));
  }, []);

  useEffect(() => {
    if (!ep) return;
    fetch(`/api/cues?ep=${encodeURIComponent(ep)}`).then((r) => r.json()).then((c: Cue[]) => { setCues(c); setSel(0); setStatus(""); })
      .catch(() => setStatus("Cannot load episode"));
  }, [ep]);

  const checks = useMemo(() => cues.map(check), [cues]);
  const visible = useMemo(() => cues.filter((c) => !onlyFlagged || checks[c.i].flags.length), [cues, checks, onlyFlagged]);
  const cue = cues[sel];
  useEffect(() => { setDraft(cue?.en ?? ""); }, [cue?.i, cue?.en]);
  useEffect(() => { listRef.current?.querySelector(`[data-i="${sel}"]`)?.scrollIntoView({ block: "nearest" }); }, [sel]);

  const move = useCallback((d: number) => {
    const at = visible.findIndex((c) => c.i === sel);
    const next = visible[Math.min(Math.max(at + d, 0), visible.length - 1)];
    if (next) setSel(next.i);
  }, [visible, sel]);

  const save = useCallback(async (andNext: boolean) => {
    if (!cue) return;
    if (draft !== cue.en) {
      setStatus("Saving…");
      const r = await fetch(`/api/cues/${cue.i}?ep=${encodeURIComponent(ep)}`, {
        method: "PUT", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ en: draft }),
      });
      if (!r.ok) { setStatus("Save failed"); return; }
      setCues((cs) => cs.map((c) => (c.i === cue.i ? { ...c, en: draft } : c)));
      setStatus(`Saved cue ${cue.i + 1}`);
    }
    if (andNext) move(1);
  }, [cue, draft, ep, move]);

  const onKey = (e: React.KeyboardEvent) => {
    if (e.ctrlKey && e.key === "Enter") { e.preventDefault(); void save(true); }
    else if (e.ctrlKey && e.key === "ArrowDown") { e.preventDefault(); move(1); }
    else if (e.ctrlKey && e.key === "ArrowUp") { e.preventDefault(); move(-1); }
  };

  const m = cue ? check({ ...cue, en: draft }) : null;
  const flagged = checks.filter((c) => c.flags.length).length;

  return (
    <div className="app" onKeyDown={onKey}>
      <header className="top">
        <h1>Subtitle Review</h1>
        <label htmlFor="ep" style={{ position: "absolute", left: -9999 }}>Episode</label>
        <select id="ep" value={ep} onChange={(e) => setEp(e.target.value)}>
          {episodes.map((e) => <option key={e} value={e}>{e}</option>)}
        </select>
        <button aria-pressed={onlyFlagged} onClick={() => setOnlyFlagged((v) => !v)}>Flagged only ({flagged})</button>
        <span className="status" role="status" aria-live="polite">{status || `${cues.length} cues`}</span>
      </header>
      <div className="main">
        <div className="list" ref={listRef} role="listbox" aria-label="Cues">
          {visible.length === 0 && <div className="empty">{cues.length ? "No flagged cues." : "No episode loaded."}</div>}
          {visible.map((c) => {
            const f = checks[c.i].flags;
            return (
              <button key={c.i} data-i={c.i} className="row" role="option" aria-selected={c.i === sel} onClick={() => setSel(c.i)}>
                <span className="mono chip">{c.i + 1}</span>
                <span className="mono time">{tc(c.start)}</span>
                <span className="tr" lang="tr">{c.tr}</span>
                <span className="en" lang="en">{c.en}</span>
                <span className={`chip ${f.includes("Empty") ? "bad" : "warn"}`}>{f[0] ?? ""}</span>
              </button>
            );
          })}
        </div>
        <section className="editor" aria-label="Cue editor">
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
                <button className="primary" onClick={() => void save(true)}>Save and next</button>
                <button onClick={() => setDraft(cue.en)} disabled={draft === cue.en}>Revert</button>
              </div>
              <div className="hint">Ctrl+Enter save and next · Ctrl+↑/↓ previous/next cue</div>
            </>
          ) : <div className="empty">Pick an episode.</div>}
        </section>
      </div>
    </div>
  );
}
