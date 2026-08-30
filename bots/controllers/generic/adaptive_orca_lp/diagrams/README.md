# Adaptive Orca LP — diagrams

Six flowcharts covering the strategy, in three formats each.

| # | Diagram | Shows |
|---|---------|-------|
| 01 | `01-architecture` | The pipeline: pool price → volatility → regime → dynamic range → economic filter → LPExecutor |
| 02 | `02-control-cycle` | The decision tree run on every control tick |
| 03 | `03-rebalance-gates` | The four gates between "near the edge" and "submit a transaction" |
| 04 | `04-regimes` | CALM / NORMAL / HIGH_VOL / EXTREME → range multiplier → width formula |
| 05 | `05-state-machine` | Controller states and every transition between them |
| 06 | `06-funding-and-recovery` | Autoswap funding, and what a restart does with an existing position |

`00-adaptive-orca-lp-all.excalidraw` is all six stacked on one canvas.

## Formats

- **`.excalidraw`** — import at [excalidraw.com](https://excalidraw.com) via *Open* (or drag the
  file onto the canvas) to edit. Everything is a real shape with bound text, so boxes stay
  attached to their labels when you move them.
- **`.svg`** — vector, for slides and docs.
- **`.png`** — 2x raster, for the submission form and GitHub.

## Regenerating

All three formats are emitted from a single layout spec, so they never drift apart:

```
scratchpad/diagrams_spec.py    # nodes, edges, routing, auto-fit
scratchpad/emit_excalidraw.py  # -> .excalidraw
scratchpad/emit_images.py      # -> .svg + .png
```

Fonts differ by format: the `.excalidraw` files use Excalidraw's hand-drawn face, while the
`.svg`/`.png` fall back to whatever comic/handwriting font the renderer has. Geometry and
content are identical.
