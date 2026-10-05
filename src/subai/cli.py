"""Command line interface: `python -m subai ...`"""
import logging
import os
from pathlib import Path
from typing import List, Optional

import typer

from subai import __version__
from subai.glossary import DEFAULT_DIR as DEFAULT_GLOSSARY_DIR
from subai.logs import setup_logging

app = typer.Typer(add_completion=False, help="Local Turkish transcription to SRT subtitles.")
log = logging.getLogger("subai")

DEFAULT_MODEL = os.environ.get("SUBAI_MODEL", "large-v3")


@app.command()
def version() -> None:
    """Print version."""
    typer.echo(__version__)


@app.command()
def run(
    input: Optional[List[Path]] = typer.Option(None, "--input", "-i", help="Media file(s). Repeatable."),
    input_dir: Optional[Path] = typer.Option(
        None, "--input-dir", envvar="SUBAI_INPUT_DIR", help="Folder of media files (batch). Default /input if no --input."
    ),
    batch: bool = typer.Option(False, "--batch", help="Process every media file in --input-dir (default /input)."),
    output_dir: Path = typer.Option(Path("/output"), "--output-dir", "-o", envvar="SUBAI_OUTPUT_DIR"),
    lang: str = typer.Option("tr", "--lang", help="Spoken language code."),
    model: str = typer.Option(DEFAULT_MODEL, "--model", help="Whisper model (large-v3, large-v3-turbo, medium, small...)."),
    device: str = typer.Option("auto", "--device", help="auto | cuda | cpu"),
    compute_type: str = typer.Option("auto", "--compute-type", help="auto | int8_float16 | int8 | float16"),
    audio_track: Optional[int] = typer.Option(None, "--audio-track", help="0-based audio stream index."),
    series: Optional[str] = typer.Option(
        None, "--series", help="Series glossary id, e.g. tvdb-383383. Auto-detected from a {tvdb-ID} folder in the input path."
    ),
    glossary_dir: Path = typer.Option(DEFAULT_GLOSSARY_DIR, "--glossary-dir", envvar="SUBAI_GLOSSARY_DIR"),
    glossary_prompt: bool = typer.Option(
        False, "--glossary-prompt", help="Prime Whisper with character names from the series glossary (experimental)."
    ),
    initial_prompt: Optional[str] = typer.Option(
        None, "--initial-prompt", help='Names/terms to bias spelling, e.g. "Evren Bey, Serkan, Peri Hanım".'
    ),
    recursive: bool = typer.Option(False, "--recursive", "-r", help="Search --input-dir recursively."),
    overwrite: bool = typer.Option(False, "--overwrite", help="Replace existing .srt outputs (re-uses cached transcription)."),
    retranscribe: bool = typer.Option(False, "--retranscribe", help="Ignore the cached transcription and run Whisper again."),
    diarize: bool = typer.Option(False, "--diarize", help="Detect speakers and write dialogue dashes (needs HF_TOKEN once; GPU)."),
    no_tone: bool = typer.Option(False, "--no-tone", help="Plain punctuation: skip continuation dots and interjection '!'."),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Transcribe media to <name>.<lang>.srt. Single file: -i FILE. Batch: --input-dir DIR."""
    from subai.pipeline.runner import discover, run_batch
    from subai.pipeline.transcribe import Transcriber

    setup_logging(verbose)
    if not input and input_dir is None:
        input_dir = Path("/input")
    files = discover(list(input or []), input_dir, recursive)
    if not files:
        log.error("No media files found. Check the path and file extensions.")
        raise typer.Exit(2)
    log.info("Found %d file(s)", len(files))

    from subai.glossary import GlossaryError, asr_prompt, detect_series_id, load_language, load_series

    try:
        language = load_language(glossary_dir, lang)
        hallucinations = language.get("asr_hallucinations") or []
        tone = None if no_tone else language.get("tone")
    except (GlossaryError, OSError) as exc:
        log.warning("No language glossary for %r (%s); hallucination filter and tone rules off", lang, exc)
        hallucinations, tone = [], None

    sid = series or next(filter(None, (detect_series_id(f) for f, _ in files)), None)
    sg = None
    if sid:
        try:
            sg = load_series(glossary_dir, sid)
        except GlossaryError as exc:
            log.error("Glossary error: %s", exc)
            raise typer.Exit(2)
        if sg is None:
            log.warning("No series glossary for %s in %s", sid, glossary_dir)
        else:
            log.info("Series glossary: %s (%s, %d characters)", sg.title.get("original"), sid, len(sg.characters))
            if glossary_prompt and not initial_prompt:
                initial_prompt = asr_prompt(sg)
                log.info("Whisper prompt from glossary: %s", initial_prompt)

    tr = Transcriber(model=model, device=device, compute_type=compute_type, lang=lang,
                     initial_prompt=initial_prompt)
    results = run_batch(files, output_dir, tr, lang, overwrite, audio_track,
                        sg.corrections if sg else None, retranscribe, hallucinations, tone, diarize)

    ok = sum(r.status == "ok" for r in results)
    skipped = sum(r.status == "skipped" for r in results)
    failed = [r for r in results if r.status == "failed"]
    log.info("Done: %d ok, %d skipped, %d failed", ok, skipped, len(failed))
    for r in failed:
        log.error("  %s: %s", r.input, r.error)
    raise typer.Exit(1 if failed else 0)


@app.command()
def translate(
    input: Path = typer.Option(..., "--input", "-i", help="Turkish .srt"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Default: <input> with .tr.srt -> .en.srt"),
    series: Optional[str] = typer.Option(None, "--series", help="Series glossary id, e.g. tvdb-383383."),
    glossary_dir: Path = typer.Option(DEFAULT_GLOSSARY_DIR, "--glossary-dir", envvar="SUBAI_GLOSSARY_DIR"),
    limit: Optional[int] = typer.Option(None, "--limit", help="Only the first N cues (for trials)."),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Translate a Turkish SRT to English with opus-mt plus glossary name protection (GPU, no LLM)."""
    from subai.glossary import GlossaryError, load_language, load_series
    from subai.translate import build_phrase_map, translate_srt

    setup_logging(verbose)
    sg = load_series(glossary_dir, series) if series else None
    dst = output or input.with_name(input.name.replace(".tr.srt", ".en.srt") if ".tr.srt" in input.name else input.stem + ".en.srt")
    try:
        phrases = build_phrase_map(load_language(glossary_dir, "tr"))
    except (OSError, GlossaryError) as exc:
        log.warning("no phrase map (%s)", exc)
        phrases = {}
    n = translate_srt(input, dst, sg, limit, phrase_map=phrases)
    log.info("Wrote %s (%d cues)", dst, n)


@app.command("retime")
def retime_cmd(
    input: Path = typer.Option(..., "--input", "-i", help="Turkish .srt with the right text but wrong times."),
    video: Path = typer.Option(..., "--video", help="The video the subtitle belongs to."),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Default: <input name>.retimed.srt beside the input."),
    lang: str = typer.Option("tr", "--lang", help="Subtitle language code."),
    model: str = typer.Option(DEFAULT_MODEL, "--model", help="Whisper model, used only when the video has no cached transcript."),
    search_root: Path = typer.Option(Path("/output"), "--search-root", envvar="SUBAI_OUTPUT_DIR",
                                     help="Where cached transcripts (.subai/<video>.words.json) are looked up."),
    audio_track: Optional[int] = typer.Option(None, "--audio-track", help="0-based audio stream index."),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Move a subtitle's cues onto the video's audio (text is never changed). Exit 2 if the evidence is too thin."""
    from subai.pipeline.runner import retime_file
    from subai.pipeline.transcribe import Transcriber
    from subai.retime import RetimeRefused

    setup_logging(verbose)
    dst = output or input.with_name(input.stem + ".retimed.srt")
    try:
        report, source = retime_file(video, input, dst, lambda: Transcriber(model=model, lang=lang), search_root, lang, audio_track)
    except RetimeRefused as exc:
        log.error("Refused, nothing written: %s", exc)
        raise typer.Exit(2)
    log.info("Transcript: %s. %d of %d cues anchored; %s; residual median %.2f s, p95 %.2f s",
             source, report.anchored_cues, report.cues, report.method, report.residual_p50, report.residual_p95)
    for p in report.pieces:
        log.info("  %.0f-%.0f s: offset %+.2f s%s", p.t0, p.t1, p.offset0, f", drift {p.slope * 1000:+.1f} ms/s" if p.slope else "")
    log.info("Wrote %s", dst)


@app.command("download-model")
def download_model_cmd(model: str = typer.Argument(DEFAULT_MODEL), verbose: bool = False) -> None:
    """Download a Whisper model into the model cache (needs network once)."""
    from faster_whisper.utils import download_model

    setup_logging(verbose)
    log.info("Downloading %s ...", model)
    try:
        path = download_model(model)
    except Exception as exc:
        log.error("Download failed: %s. Check your network connection and free disk space.", exc)
        raise typer.Exit(1)
    log.info("Model ready at %s", path)


@app.command()
def doctor(model: str = typer.Option(DEFAULT_MODEL, "--model")) -> None:
    """Check ffmpeg, GPU, disk, model files; run a smoke test."""
    from subai.doctor import run_doctor

    setup_logging(False)
    raise typer.Exit(0 if run_doctor(model, Path("/output")) else 1)


@app.command("glossary-fetch")
def glossary_fetch(
    series: str = typer.Option("tvdb-383383", "--series", help="Series id, e.g. tvdb-383383."),
    glossary_dir: Path = typer.Option(DEFAULT_GLOSSARY_DIR, "--glossary-dir", envvar="SUBAI_GLOSSARY_DIR"),
    write: bool = typer.Option(False, "--write", help="Write fetched.yaml and episodes.yaml (default: dry run)."),
) -> None:
    """Refresh series metadata from TMDB + TVDB (needs TMDB_API_KEY and TVDB_API_KEY; uses the network)."""
    from subai.glossary import find_series_dir
    from subai.glossary_fetch import FetchError, run_fetch

    setup_logging(False)
    d = find_series_dir(glossary_dir, series)
    if d is None:
        log.error("No series glossary folder for %s in %s", series, glossary_dir)
        raise typer.Exit(2)
    tmdb_key, tvdb_key = os.environ.get("TMDB_API_KEY", ""), os.environ.get("TVDB_API_KEY", "")
    if not tmdb_key or not tvdb_key:
        log.error("Set TMDB_API_KEY and TVDB_API_KEY (docker/.env).")
        raise typer.Exit(2)
    try:
        out = run_fetch(d, tmdb_key, tvdb_key, write)
    except FetchError as exc:
        log.error("Fetch failed: %s", exc)
        raise typer.Exit(1)
    f, eps = out["fetched"], out["episodes"]["episodes"]
    diff = f["diff_vs_characters_yaml"]
    log.info("TMDB: %d cast roles, %d episodes, %d alternative titles | TVDB: %d characters, %d aliases",
             len(f["tmdb"]["cast"]), len(eps), len(f["tmdb"]["alternative_titles"]),
             len(f["tvdb"]["characters"]), len(f["tvdb"]["aliases"]))
    log.info("Overviews: %d tr, %d en | guest stars listed in %d episodes",
             sum(bool(e["overview_tr"]) for e in eps), sum(bool(e["overview_en"]) for e in eps),
             sum(bool(e["guest_stars"]) for e in eps))
    for key, rows in diff.items():
        log.info("%s: %d", key, len(rows))
        for r in rows:
            log.info("   %s", r)
    log.info("Wrote fetched.yaml and episodes.yaml in %s" % d if write else "Dry run (use --write to save).")
