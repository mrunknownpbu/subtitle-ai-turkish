# Design System — Review Editor

Utility-first, dense, keyboard-driven tool for long editing sessions. Calm, low-fatigue visuals; content (text, waveform) is the hero.

## 1. Principles
1. **Focus on the line.** The current cue is always visually dominant.
2. **Keyboard first.** Every frequent action has a shortcut; mouse is optional.
3. **Dense but legible.** Many cues visible; never sacrifice contrast or text size.
4. **Status is explicit.** Flags and errors use icon + text, never color alone.
5. **Dark by default**, light theme supported.

## 2. Color Tokens
Define as CSS variables on `:root`; override in `[data-theme="light"]`.

| Token | Dark | Light | Use |
|---|---|---|---|
| `--bg` | #0F1115 | #FAFAF9 | App background |
| `--surface` | #171A21 | #FFFFFF | Panels |
| `--surface-2` | #1F232C | #F1F0EE | Rows, inputs |
| `--border` | #2A2F3A | #DDDAD5 | Dividers |
| `--text` | #E7E9EE | #1A1C21 | Primary text |
| `--text-muted` | #9AA1AF | #5E6470 | Secondary |
| `--accent` | #5B9DFF | #1F6FEB | Selection, focus, primary action |
| `--success` | #4CC38A | #1A7F4B | Passing check |
| `--warning` | #F2B84B | #8A5A00 | Needs review |
| `--danger` | #FF6B6B | #C62828 | Rule violation |
| `--waveform` | #5B9DFF | #1F6FEB | Waveform |
| `--cue-block` | accent @ 25% | accent @ 18% | Cue regions on waveform |

All text/background pairs ≥ 4.5:1 (WCAG AA).

## 3. Typography
- UI: Inter, system-ui fallback. Subtitle text: Inter or Noto Sans (full Turkish glyph support: ç ğ ı İ ö ş ü).
- Mono (timecodes, CPS): JetBrains Mono, tabular numerals.

| Role | Size / Line | Weight |
|---|---|---|
| Cue text (editing) | 16 / 24 | 400 |
| UI body | 14 / 20 | 400 |
| Label / meta | 12 / 16 | 500 |
| Panel title | 14 / 20 | 600 |
| Timecode | 13 / 16 mono | 500 |

## 4. Spacing, Radius, Elevation
- 4 px base grid: 4, 8, 12, 16, 24, 32.
- Radius: 6 px controls, 10 px panels.
- Elevation via border + subtle shadow only on popovers/menus. No heavy shadows.

## 5. Layout
```
┌────────────────────────────────────────────────────────┐
│ Top bar: episode · target lang · status · Export       │
├───────────────────────────┬────────────────────────────┤
│ Video player              │ Cue editor (current cue)   │
│                           │  TR source │ Target text   │
├───────────────────────────┴────────────────────────────┤
│ Waveform + cue regions + playhead                      │
├────────────────────────────────────────────────────────┤
│ Cue list (virtualized): # · time · TR · target · flags │
└────────────────────────────────────────────────────────┘
```
Right-side drawer (toggle): Glossary · Characters · Flags queue.
Responsive: below 1024 px, stack player above editor; cue list becomes full-screen view.

## 6. Components
| Component | Notes |
|---|---|
| **CueRow** | Index, start–end, TR text (muted), target text, CPS chip, flag icons. States: default, hover, selected, editing, error |
| **CueEditor** | Two textareas side by side, live char count per line, CPS meter, split/merge buttons |
| **Waveform** | wavesurfer.js, draggable cue edges with snap to word boundaries, zoom |
| **TimeInput** | `HH:MM:SS,mmm` mask, arrow keys nudge ±40 ms (Shift ±1 s) |
| **FlagBadge** | Icon + label: Idiom, Low confidence, Unknown name, Gender?, Too long |
| **RuleMeter** | Characters/line and CPS bars; turns warning/danger with text label |
| **GlossaryPanel** | Term list, inline add, "apply to all matching cues" |
| **ProgressStepper** | Pipeline stages with status and elapsed time |
| **Toast** | Autosave confirmations, non-blocking errors |

## 7. Interaction & Shortcuts
| Key | Action |
|---|---|
| Space | Play/pause (when not typing) |
| Ctrl+Enter | Save cue and go to next |
| Ctrl+↑ / ↓ | Previous / next cue |
| Ctrl+Shift+F | Next flagged cue |
| Ctrl+L | Replay current cue |
| Ctrl+K | Split cue at caret |
| Ctrl+J | Merge with next |
| Alt+← / → | Nudge start time |
| Ctrl+Z / Y | Undo / redo |
| Ctrl+E | Export dialog |
| ? | Shortcut cheatsheet |

## 8. Motion
150 ms ease-out for hover/focus/panel transitions. Respect `prefers-reduced-motion` (disable animation, keep state changes).

## 9. Accessibility
- Visible 2 px focus ring in `--accent`, offset 2 px.
- All controls reachable by keyboard; logical tab order; ARIA labels for icon buttons.
- Live region announces autosave and validation results.
- Lang attributes set per text field (`tr`, `en`, `ms`, …) for correct screen reader pronunciation.

## 10. Content & Copy
- Plain, short verbs: "Export", "Re-translate", "Apply glossary".
- Errors state cause and fix: "Line 2 has 47 characters (max 42). Split the cue or shorten the text."
- Subtitle preview shows rendered style (white text, dark outline) on the video at 100% scale.

## 11. Subtitle Preview Style (ASS defaults)
Font Noto Sans 44, white fill, black 2 px outline, bottom-center, margin 40 px.

## 12. Implementation Notes
- Tokens in `web/src/styles/tokens.css`; Tailwind config reads the same variables.
- Component library: Radix UI primitives + own styling; icons: lucide.
- Virtualize cue list (≥ 1,000 rows typical per episode).
