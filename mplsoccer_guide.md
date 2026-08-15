# mplsoccer overview — pass maps & heatmaps

Reference notes for building pitch visualizations (pass maps, flow maps, heatmaps) in
`Pre_Match_Report.py` using [mplsoccer](https://mplsoccer.readthedocs.io/), a matplotlib
wrapper for pitch plots. Since `Pre_Match_Report.py` already builds on `plt` / `PdfPages`,
mplsoccer figures drop straight into the existing report pipeline as just another figure.

Install: `pip install mplsoccer` (not yet in `requirements.txt` — that file is currently empty).

## 1. The `Pitch` object

Everything starts here — it draws the pitch and gives you plotting methods that take raw
x/y coordinates instead of you hand-computing SVG paths.

```python
from mplsoccer import Pitch

pitch = Pitch(pitch_type='opta', pitch_color='#f4f4f2', line_color='#888888', linewidth=1)
fig, ax = pitch.draw(figsize=(10, 7))
```

**Coordinate system — check this first.** `pitch_type='opta'` assumes 0–100 on both axes,
which is the closest built-in match to SofaScore's normalized percentage coordinates. But
"closest match" isn't "guaranteed match" — before trusting any plot, render a few known
points (e.g. the GK's average position, which should land near his own goal) and confirm
nothing's mirrored or flipped. If y comes out inverted, flip it yourself (`100 - y`) before
plotting rather than fighting the library's orientation assumptions.

## 2. Pass map (arrows)

`pitch.arrows()` — draws one arrow per pass, using `Player_Event_Data.csv` columns directly:

```python
import pandas as pd

df = pd.read_csv('Player_Event_Data.csv')
gk = df[(df['Player'] == 'Nicholas Hagen') & (df['Event Type'] == 'pass')]

pitch.arrows(
    gk['Player X Coord'], gk['Player Y Coord'],
    gk['Pass End X Coord'], gk['Pass End Y Coord'],
    ax=ax, color='#2a78d6', width=1.5, headwidth=6, headlength=6, alpha=0.75
)
```

- Color by outcome or length bucket by passing an array instead of one color:
  `color=gk['Outcome'].map({True: '#0ca30c', False: '#d03b3b'})` is the standard
  "green = completed, red = turnover" pass-map convention.
- `width` also accepts an array, so line thickness can scale with a per-row value if
  you've pre-aggregated (see below) rather than plotting every raw pass at equal weight.

## 3. Zone-to-zone flow map (aggregated)

`arrows()` only draws what you give it — mplsoccer has no built-in "aggregate to zones"
step, so that's done in pandas first:

```python
def zone(x, bins, labels):
    return pd.cut(x, bins=bins, labels=labels)

gk['origin_zone'] = zone(gk['Player Y Coord'], bins=[0, 20, 40, 60, 80, 100], labels=range(5))
gk['dest_zone']   = zone(gk['Pass End Y Coord'], bins=[0, 20, 40, 60, 80, 100], labels=range(5))

agg = gk.groupby(['origin_zone', 'dest_zone']).size().reset_index(name='count')
# join zone centroid x/y onto agg as x1,y1 (origin) / x2,y2 (dest), then:
pitch.arrows(agg['x1'], agg['y1'], agg['x2'], agg['y2'],
             ax=ax, width=agg['count'] / agg['count'].max() * 8, color='#2a78d6')
```

## 4. Heatmaps

Two options, same `Pitch` object:

**Discrete grid** (best when you want a clean "which zone" story):
```python
bin_stat = pitch.bin_statistic(gk['Player X Coord'], gk['Player Y Coord'],
                                statistic='count', bins=(4, 5))
pitch.heatmap(bin_stat, ax=ax, cmap='Blues', edgecolor='#f4f4f2')
pitch.label_heatmap(bin_stat, ax=ax, color='white', fontsize=8, ha='center', va='center')
```
`bins` also accepts explicit edge arrays, so the grid can be forced onto exact
third-boundaries instead of even splits. `statistic` can be `'mean'` / `'sum'` over a
`values=` array too — e.g. average pass length per zone instead of count.

**Smooth KDE** (needs `seaborn` installed under the hood; better for exploratory looks,
blurrier for a printed report):
```python
pitch.kdeplot(gk['Player X Coord'], gk['Player Y Coord'], ax=ax, cmap='Blues', fill=True, thresh=0)
```

Stick to one hue ramp (`'Blues'` or similar) for magnitude — sequential data = one color,
light→dark, not a rainbow colormap like `'jet'`.

## 5. Cropping to the relevant area

Build-out only touches the defensive + middle thirds — don't draw a full pitch with an
empty attacking third, just crop the axes after drawing:

```python
fig, ax = pitch.draw(figsize=(10, 7))
ax.set_xlim(0, 70)
```

(More reliable than mplsoccer's own `half=True` option, which assumes a specific attacking
direction that may not match SofaScore's convention.)

## 6. Dropping it into the existing report

Same pattern `Pre_Match_Report.py` already uses for other figures:

```python
with PdfPages(output_path) as pdf:
    fig, ax = pitch.draw(figsize=(10, 7))
    pitch.arrows(..., ax=ax)
    pdf.savefig(fig)
    plt.close(fig)
```

## Related context

- `Player_Event_Data.csv` has no receiver field or timestamp — pass-chaining
  ("who received it, what did they do next") isn't derivable from this data as scraped.
  See project memory for the full breakdown.
- For a GK-only pass map, the filter that matters is `Event Type == 'pass'`; an added
  "defensive third only" filter is redundant since GK passes already originate there.