import { useEffect, useRef, useState } from "react";

/** Native modal <dialog>: focus trap and Esc come for free. */
export function Modal({ title, onClose, children }: { title: string; onClose: () => void; children: React.ReactNode }) {
  const ref = useRef<HTMLDialogElement>(null);
  useEffect(() => { ref.current?.showModal(); }, []);
  return (
    <dialog ref={ref} onClose={onClose} aria-label={title}>
      <h2>{title}</h2>
      {children}
    </dialog>
  );
}

const err = async (r: Response) => ((await r.json().catch(() => ({}))) as { detail?: string }).detail ?? `Error ${r.status}`;
const enc = encodeURIComponent;
const base = (p: string) => p.split("/").pop() ?? p;

/** Walk the media library. onPick gets the file's path under the library; it returns an error message to keep the dialog open. */
function FilePicker({ title, kind, start, hint, emptyLabel, onPick, onClose }: {
  title: string; kind: "video" | "srt"; start: string; hint: string; emptyLabel?: string;
  onPick: (path: string) => Promise<string | void>; onClose: () => void;
}) {
  const [path, setPath] = useState(start.includes("/") ? start.slice(0, start.lastIndexOf("/")) : "");
  const [list, setList] = useState<{ dirs: string[]; files: string[] }>({ dirs: [], files: [] });
  const [msg, setMsg] = useState("");

  useEffect(() => {
    setMsg("");
    fetch(`/api/browse?path=${enc(path)}&kind=${kind}`).then((r) => (r.ok ? r.json() : Promise.reject()))
      .then(setList).catch(() => { setList({ dirs: [], files: [] }); setMsg("Cannot open this folder"); });
  }, [path, kind]);

  const pick = async (file: string) => {
    const e = await onPick(file);
    if (e) setMsg(e); else onClose();
  };

  const parts = path ? path.split("/") : [];
  const join = (n: string) => (path ? `${path}/${n}` : n);
  return (
    <Modal title={title} onClose={onClose}>
      <nav className="crumbs" aria-label="Folder">
        <button onClick={() => setPath("")}>Library</button>
        {parts.map((p, i) => <button key={i} onClick={() => setPath(parts.slice(0, i + 1).join("/"))}>{p}</button>)}
      </nav>
      <div className="browse">
        {path && <button onClick={() => setPath(parts.slice(0, -1).join("/"))}>.. (up)</button>}
        {list.dirs.map((d) => <button key={d} onClick={() => setPath(join(d))}>▸ {d}</button>)}
        {list.files.map((f) => <button key={f} className="file" onClick={() => void pick(join(f))}>{f}</button>)}
        {!list.dirs.length && !list.files.length && <div className="empty">No folders or files here.</div>}
      </div>
      <div className="hint" role="status">{msg || hint}</div>
      <div className="actions">
        {emptyLabel && <button onClick={() => void pick("")}>{emptyLabel}</button>}
        <button onClick={onClose}>Cancel</button>
      </div>
    </Modal>
  );
}

/** Pick the video for the current episode. */
export function VideoPicker({ ep, current, onClose, onChosen }: { ep: string; current: string; onClose: () => void; onChosen: () => void }) {
  const choose = async (file: string) => {
    const r = await fetch(`/api/video-choice?ep=${enc(ep)}`, {
      method: "PUT", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ file }),
    });
    if (!r.ok) return await err(r);
    onChosen();
  };
  return (
    <FilePicker title="Choose video" kind="video" start={current} emptyLabel="Use automatic match" onPick={choose} onClose={onClose}
      hint="MKV and MP4. MKV is converted for the browser on first play." />
  );
}

const stem = (f: string) => f.replace(/\.srt$/i, "").replace(/\.(tr|en)(\.hi)?$/i, "");

/** Upload a Turkish and/or English SRT as a new episode. The missing language starts blank with the same timings. */
export function UploadDialog({ existing, onClose, onDone }: { existing: string[]; onClose: () => void; onDone: (ep: string) => void }) {
  const [name, setName] = useState("");
  const [files, setFiles] = useState<{ tr?: File; en?: File }>({});
  const [msg, setMsg] = useState("");
  const [busy, setBusy] = useState(false);

  const pick = (lang: "tr" | "en", f?: File) => {
    setFiles((s) => ({ ...s, [lang]: f }));
    if (f && !name) setName(stem(f.name));
  };

  const [replace, setReplace] = useState(false);

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    const n = name.trim();
    if (!n || (!files.tr && !files.en)) { setMsg("Enter a name and choose at least one SRT file."); return; }
    setBusy(true); setMsg("Uploading…");
    let ep = "";
    for (const lang of ["tr", "en"] as const) {
      const f = files[lang];
      if (!f) continue;
      const r = await fetch(`/api/upload?name=${enc(n)}&lang=${lang}&replace=${replace}`, { method: "POST", body: f });
      if (!r.ok) { setMsg(`${f.name}: ${await err(r)}`); setBusy(false); return; }
      ep = ((await r.json()) as { ep: string }).ep;
    }
    onDone(ep); onClose();
  };

  return (
    <Modal title="Upload subtitles" onClose={onClose}>
      <form onSubmit={(e) => void submit(e)} className="upload">
        <label htmlFor="up-name">Episode name</label>
        <input id="up-name" value={name} onChange={(e) => setName(e.target.value)} />
        <label htmlFor="up-tr">Turkish SRT</label>
        <input id="up-tr" type="file" accept=".srt" onChange={(e) => pick("tr", e.target.files?.[0])} />
        <label htmlFor="up-en">English SRT</label>
        <input id="up-en" type="file" accept=".srt" onChange={(e) => pick("en", e.target.files?.[0])} />
        <label className="check"><input type="checkbox" checked={replace} onChange={(e) => setReplace(e.target.checked)} /> Replace the subtitles if this name already exists</label>
        <div className="hint" role="status">{msg || "Choose one or both. A missing language starts empty with the same timings. Then pick its video with Change video."}</div>
        <div className="actions">
          <button className="primary" type="submit" disabled={busy}>Upload</button>
          <button type="button" onClick={onClose}>Cancel</button>
        </div>
      </form>
    </Modal>
  );
}

/** Submit a job. It joins the queue and runs in order; the dialog closes and the Jobs list shows progress. */
function useSubmit(onQueued: () => void, onClose: () => void) {
  return async (r: Response) => {
    if (!r.ok) return await err(r);
    onQueued(); onClose();
    return "";
  };
}

type Source = { mode: "upload" | "library"; file?: File; lib: string };

/** Upload an SRT from this computer, or choose one from the media library. */
function SourceChooser({ label, startAt, source, setSource, disabled, onName }: {
  label: string; startAt: string; source: Source; setSource: (s: Source) => void; disabled: boolean; onName: (n: string) => void;
}) {
  const [open, setOpen] = useState(false);
  return (
    <>
      <label>{label}</label>
      <div className="actions" role="tablist">
        <button role="tab" aria-selected={source.mode === "upload"} onClick={() => setSource({ ...source, mode: "upload" })} disabled={disabled}>Upload</button>
        <button role="tab" aria-selected={source.mode === "library"} onClick={() => setSource({ ...source, mode: "library" })} disabled={disabled}>Choose from library</button>
      </div>
      {source.mode === "upload" ? (
        <input type="file" accept=".srt" aria-label={label} disabled={disabled}
          onChange={(e) => { const f = e.target.files?.[0]; setSource({ ...source, file: f }); if (f) onName(f.name); }} />
      ) : (
        <div className="vfile">
          <span className="hint" title={source.lib}>{source.lib ? base(source.lib) : "None chosen"}</span>
          <button onClick={() => setOpen(true)} disabled={disabled}>Browse…</button>
        </div>
      )}
      {open && (
        <FilePicker title={label} kind="srt" start={startAt || source.lib} hint="A .srt file from the library." onClose={() => setOpen(false)}
          onPick={async (p) => { setSource({ ...source, lib: p }); onName(base(p)); }} />
      )}
    </>
  );
}

function VideoChooser({ video, setVideo, disabled }: { video: string; setVideo: (v: string) => void; disabled: boolean }) {
  const [open, setOpen] = useState(false);
  return (
    <>
      <label>Video</label>
      <div className="vfile">
        <span className="hint" title={video}>{video ? base(video) : "None chosen"}</span>
        <button onClick={() => setOpen(true)} disabled={disabled}>Choose video…</button>
      </div>
      {open && <FilePicker title="Choose video" kind="video" start={video} hint="MKV and MP4." onClose={() => setOpen(false)} onPick={async (p) => { setVideo(p); }} />}
    </>
  );
}

/** Re-time an original Turkish SRT (uploaded or from the library) onto a video's audio; the result opens as a new episode. */
export function RetimeDialog({ defaultVideo, existing, onClose, onQueued }: {
  defaultVideo: string; existing: string[]; onClose: () => void; onQueued: () => void;
}) {
  const [video, setVideo] = useState(defaultVideo);
  const [source, setSource] = useState<Source>({ mode: "upload", lib: "" });
  const [name, setName] = useState("");
  const [replace, setReplace] = useState(false);
  const [msg, setMsg] = useState("");
  const begin = useSubmit(onQueued, onClose);

  const start = async () => {
    const n = name.trim();
    if (!video) { setMsg("Choose the video first."); return; }
    if (!n || (source.mode === "upload" ? !source.file : !source.lib)) { setMsg("Choose the subtitle and give the episode a name."); return; }
    setMsg(await begin(await fetch(`/api/retime?video=${enc(video)}&name=${enc(n)}&replace_original=${replace}${source.mode === "library" ? `&src=${enc(source.lib)}` : ""}`,
      { method: "POST", body: source.mode === "upload" ? source.file : undefined })));
  };

  return (
    <Modal title="Re-time Turkish subtitle" onClose={onClose}>
      <div className="upload">
        <VideoChooser video={video} setVideo={setVideo} disabled={false} />
        <SourceChooser label="Original Turkish subtitle" startAt={video} source={source} setSource={setSource} disabled={false}
          onName={(n) => { if (!name) setName(stem(n)); }} />
        <label htmlFor="rt-name">Name for the re-timed episode</label>
        <input id="rt-name" value={name} onChange={(e) => setName(e.target.value)} />
        <label className="check"><input type="checkbox" checked={replace} onChange={(e) => setReplace(e.target.checked)} /> Replace the episode if this name already exists</label>
        <div className="hint" role="status">
          {msg || (existing.includes(`uploads/${name.trim()}`) && !replace ? `"${name.trim()}" exists: the result is saved as "${name.trim()}.retimed" and the original is kept. ` : "")
            || "Added to the queue; jobs run one at a time. Moves each cue to where it is spoken; the text is never changed. A video without a cached transcript is transcribed first (about 7 minutes on the GPU)."}
        </div>
        <div className="actions">
          <button className="primary" onClick={() => void start()}>Add to queue</button>
          <button onClick={onClose}>Cancel</button>
        </div>
      </div>
    </Modal>
  );
}

/** Choose a library video and transcribe it (Turkish), translate (English), or both. The result opens as a new episode. */
export function ProcessDialog({ defaultVideo, onClose, onQueued }: { defaultVideo: string; onClose: () => void; onQueued: () => void }) {
  const [video, setVideo] = useState(defaultVideo);
  const [from, setFrom] = useState<"transcribe" | "retime" | "as-is">("transcribe"); // where the Turkish text comes from
  const [translate, setTranslate] = useState(true);
  const [source, setSource] = useState<Source>({ mode: "upload", lib: "" });
  const [models, setModels] = useState<string[]>([]);
  const [model, setModel] = useState("");
  const [msg, setMsg] = useState("");
  const [overTr, setOverTr] = useState(false);
  const [overEn, setOverEn] = useState(false);
  const begin = useSubmit(onQueued, onClose);
  const transcribe = from === "transcribe", retime = from === "retime";

  useEffect(() => {
    fetch("/api/models").then((r) => r.json()).then((m: { models: string[] }) => { setModels(m.models); setModel(m.models[0] ?? ""); }).catch(() => {});
  }, []);

  const start = async () => {
    if (!video) { setMsg("Choose the video first."); return; }
    if (!transcribe && !retime && !translate) { setMsg("Nothing to do: choose a step."); return; }
    const q = `video=${enc(video)}&transcribe=${transcribe}&retime=${retime}&translate=${translate}&overwrite_original=${overTr}&overwrite_english=${overEn}${transcribe ? `&model=${enc(model)}` : ""}`
      + (!transcribe && source.mode === "library" && source.lib ? `&src=${enc(source.lib)}` : "");
    setMsg(await begin(await fetch(`/api/process?${q}`, { method: "POST", body: !transcribe && source.mode === "upload" ? source.file : undefined })));
  };

  return (
    <Modal title="Transcribe, re-time and translate" onClose={onClose}>
      <div className="upload">
        <VideoChooser video={video} setVideo={setVideo} disabled={false} />
        <label htmlFor="pr-from">Turkish text</label>
        <select id="pr-from" value={from} disabled={false} onChange={(e) => setFrom(e.target.value as typeof from)}>
          <option value="transcribe">Transcribe the audio</option>
          <option value="retime">Re-time a Turkish subtitle onto this video</option>
          <option value="as-is">Use a Turkish subtitle as it is (or this video's earlier one)</option>
        </select>
        {transcribe && (
          <>
            <label htmlFor="pr-model">Whisper model</label>
            <select id="pr-model" value={model} disabled={false} onChange={(e) => setModel(e.target.value)}>
              {models.map((m) => <option key={m} value={m}>{base(m)}</option>)}
            </select>
          </>
        )}
        {!transcribe && (
          <SourceChooser label={retime ? "Turkish subtitle to re-time" : "Turkish subtitle (optional: otherwise this video's earlier transcription)"}
            startAt={video} source={source} setSource={setSource} disabled={false} onName={() => {}} />
        )}
        <label className="check"><input type="checkbox" checked={translate} disabled={false} onChange={(e) => setTranslate(e.target.checked)} /> Then translate Turkish to English</label>
        <label className="check"><input type="checkbox" checked={overTr} onChange={(e) => setOverTr(e.target.checked)} /> Replace the Turkish subtitle if this video already has one</label>
        <label className="check"><input type="checkbox" checked={overEn} onChange={(e) => setOverEn(e.target.checked)} /> Replace the English subtitle if this video already has one</label>
        <div className="hint" role="status">
          {msg || "Existing subtitles are kept unless you tick Replace (the step is skipped and logged as KEEP). Jobs run one at a time. Transcribing takes about 7 minutes per episode on the GPU, re-timing about a minute, translating under a minute."}
        </div>
        <div className="actions">
          <button className="primary" onClick={() => void start()}>Add to queue</button>
          <button onClick={onClose}>Cancel</button>
        </div>
      </div>
    </Modal>
  );
}
