import { useState } from "react";
import { Modal } from "./Pickers";

export type JobRow = {
  id: string; state: string; log: string[]; ep: string; title: string; video: string;
  started: number; ended: number | null; step: string;
};

// icon + word, never colour alone
const STATE: Record<string, { icon: string; text: string; tone: string }> = {
  running: { icon: "◔", text: "Running", tone: "run" },
  done: { icon: "✓", text: "Done", tone: "ok" },
  failed: { icon: "✕", text: "Failed", tone: "bad" },
  refused: { icon: "⊘", text: "Refused", tone: "warn" },
  cancelled: { icon: "–", text: "Cancelled", tone: "muted" },
};

const TABS = [
  ["all", "All", () => true],
  ["running", "Running", (j: JobRow) => j.state === "running"],
  ["done", "Done", (j: JobRow) => j.state === "done"],
  ["problems", "Problems", (j: JobRow) => !["running", "done"].includes(j.state)],
] as const;

function ago(t: number) {
  const s = Math.max(0, Date.now() / 1000 - t);
  return s < 60 ? "just now" : s < 3600 ? `${Math.floor(s / 60)} min ago` : s < 86400 ? `${Math.floor(s / 3600)} h ago` : `${Math.floor(s / 86400)} d ago`;
}

function took(j: JobRow) {
  const s = Math.round((j.ended ?? Date.now() / 1000) - j.started);
  return s < 60 ? `${s} s` : `${Math.floor(s / 60)} min ${s % 60} s`;
}

export function JobsDialog({ jobs, onClose, onOpen, onClear, onChanged }: {
  jobs: JobRow[]; onClose: () => void; onOpen: (ep: string) => void; onClear: () => void; onChanged: () => void;
}) {
  const [tab, setTab] = useState<(typeof TABS)[number][0]>("all");
  const test = TABS.find((t) => t[0] === tab)![2];
  const shown = jobs.filter(test);
  const finished = jobs.some((j) => j.state !== "running");
  return (
    <Modal title="Jobs" onClose={onClose}>
      <div className="tabs" role="group" aria-label="Filter jobs">
        {TABS.map(([k, label, f]) => (
          <button key={k} aria-pressed={tab === k} onClick={() => setTab(k)}>{label} <span className="count">{jobs.filter(f).length}</span></button>
        ))}
      </div>
      <div className="joblist">
        {shown.length === 0 && <div className="empty">{jobs.length ? "No jobs in this view." : "No jobs yet. Start one from Tools."}</div>}
        {shown.map((j) => {
          const st = STATE[j.state] ?? STATE.failed;
          return (
            <article key={j.id} className="job">
              <div className="jobhead">
                <span className={`pill ${st.tone}`}><span aria-hidden="true">{st.icon}</span> {st.text}</span>
                <strong>{j.title}</strong>
                <span className="hint">{j.video} · {ago(j.started)} · {took(j)}{j.state === "running" && j.step ? ` · ${j.step}` : ""}</span>
              </div>
              <div className="actions">
                {j.state === "done" && <button onClick={() => { onOpen(j.ep); onClose(); }}>Open episode</button>}
                {j.state === "running" && (
                  <button onClick={() => void fetch(`/api/retime-job-cancel?id=${j.id}`, { method: "POST" }).then(onChanged)}>Cancel job</button>
                )}
              </div>
              <details>
                <summary>Log</summary>
                <pre className="joblog">{j.log.map((l) => l.trim()).join("\n") || "(empty)"}</pre>
              </details>
            </article>
          );
        })}
      </div>
      <div className="actions">
        <button onClick={onClear} disabled={!finished}>Clear finished</button>
        <button onClick={onClose}>Close</button>
      </div>
    </Modal>
  );
}

export function ShortcutsDialog({ onClose }: { onClose: () => void }) {
  const keys: [string, string][] = [
    ["Ctrl+Enter", "Save the cue and go to the next"], ["Ctrl+↑ / Ctrl+↓", "Previous / next cue"],
    ["Ctrl+L", "Replay the current cue"], ["Space", "Play or pause (when not typing)"], ["?", "This list"],
  ];
  return (
    <Modal title="Keyboard shortcuts" onClose={onClose}>
      <dl className="keys">{keys.map(([k, d]) => <div key={k}><dt className="mono">{k}</dt><dd>{d}</dd></div>)}</dl>
      <div className="actions"><button onClick={onClose}>Close</button></div>
    </Modal>
  );
}
