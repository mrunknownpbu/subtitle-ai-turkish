import { useEffect, useRef } from "react";

type Cue = { i: number; start: number; end: number };
type Props = { video: React.RefObject<HTMLVideoElement>; peaks: number[]; rate: number; cues: Cue[]; sel: number };

const WINDOW = 30; // seconds shown, playhead centred

const css = (n: string) => getComputedStyle(document.documentElement).getPropertyValue(n).trim();

/** Canvas waveform of the audio around the playhead, with cue regions. Click to seek. */
export default function Waveform({ video, peaks, rate, cues, sel }: Props) {
  const ref = useRef<HTMLCanvasElement>(null);
  const props = useRef({ peaks, rate, cues, sel });
  props.current = { peaks, rate, cues, sel };

  useEffect(() => {
    const c = ref.current!;
    let raf = 0;
    const draw = () => {
      raf = requestAnimationFrame(draw);
      const { peaks, rate, cues, sel } = props.current;
      const dpr = window.devicePixelRatio || 1;
      const w = c.clientWidth, h = c.clientHeight;
      if (c.width !== w * dpr || c.height !== h * dpr) { c.width = w * dpr; c.height = h * dpr; }
      const g = c.getContext("2d")!;
      g.setTransform(dpr, 0, 0, dpr, 0, 0);
      g.clearRect(0, 0, w, h);
      const t = video.current?.currentTime ?? 0, t0 = t - WINDOW / 2, px = w / WINDOW;
      for (const q of cues) {
        if (q.end < t0 || q.start > t0 + WINDOW) continue;
        g.fillStyle = css("--accent"); g.globalAlpha = q.i === sel ? 0.4 : 0.15;
        g.fillRect((q.start - t0) * px, 0, Math.max((q.end - q.start) * px, 1), h);
      }
      g.globalAlpha = 1; g.fillStyle = css("--accent");
      const mid = h / 2;
      for (let x = 0; x < w; x++) {
        const k = Math.floor((t0 + x / px) * rate);
        const a = k >= 0 && k < peaks.length ? (peaks[k] / 255) * mid : 0;
        g.fillRect(x, mid - a, 1, Math.max(a * 2, 1));
      }
      g.fillStyle = css("--text"); g.fillRect(w / 2 - 1, 0, 2, h);
    };
    draw();
    return () => cancelAnimationFrame(raf);
  }, [video]);

  const seek = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const r = e.currentTarget.getBoundingClientRect();
    if (video.current) video.current.currentTime = Math.max(0, (video.current.currentTime - WINDOW / 2) + ((e.clientX - r.left) / r.width) * WINDOW);
  };

  return <canvas ref={ref} className="wave" onClick={seek} role="img" aria-label="Audio waveform; click to seek" />;
}
