import pandas as pd
import numpy as np
import os
from datetime import datetime
from mplsoccer import Pitch, VerticalPitch
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib.legend import Legend
from matplotlib.markers import MarkerStyle
from matplotlib.path import Path as MarkerPath
import matplotlib.pyplot as plt

# Constants
PITCH_LENGTH_YARDS = 115
PITCH_WIDTH_YARDS = 74

# Fixed figure-fraction gap between a chart's requested bbox bottom edge (bbox[1]) and
# its notes box below it. Shared by generate_relative_gk_pass_map and
# generate_defensive_action_map, whose charts both sit on bbox[1]=0.50, so their notes
# boxes land on the same horizontal line rather than each chart computing its own offset.
NOTES_OFFSET = 0.02

# Most entries the key-pass legend will name individually before rolling the remainder
# into a single "Other" row. Bounds the legend's height the same way the Potential
# Absences table caps itself in Pre_Match_Report.generatePredictedLineup; the list is
# already sorted by key passes per 90, so the cut drops the least creative players.
# Tokens that belong to the surname rather than being a middle name, used when shortening
# names for the key-pass legend. Lowercase-compared, so it covers both "van Dijk" and the
# capitalised Arabic/Iberian patronymics.
SURNAME_PARTICLES = {
    'abou', 'abu', 'al', 'ben', 'bin', 'da', 'das', 'de', 'del', 'della', 'der', 'di',
    'do', 'dos', 'du', 'el', 'ibn', 'la', 'le', 'los', 'mac', 'mc', 'san', 'santa',
    'ten', 'ter', 'van', 'von',
}

KEY_PASS_LEGEND_MAX = 15

# A key-pass legend entry is greyed out below this share of the minutes the team actually
# played, because per-90 rates built on a cameo are noise dressed as form: measured on
# real data, a Pumas substitute with ONE key pass in 79 minutes (26% of available) scored
# 1.14/90 and outranked a near-ever-present with two in 255. Expressed as a share rather
# than a fixed minute count so it adapts to sample size on its own — 30% is ~91 minutes
# for a 3-match opponent and ~490 for an 18-match one. Sub-threshold players are still
# listed and still plotted, just visibly de-emphasised, since a rotation player can start.
KEY_PASS_MINUTES_QUALIFIED = 0.30

# Same cap-and-roll-up idea as KEY_PASS_LEGEND_MAX. A full MLS season produces 19-25 distinct
# shooters for one team, so the cap does real work; the list is sorted by npxG/90 descending, so
# the cut drops the least threatening.
#
# Held to the same 15 as KEY_PASS_LEGEND_MAX so the two player keys on the report list the same
# number of names — a reader comparing "who creates" against "who shoots" shouldn't have to
# account for the lists being different lengths. The room is there for more (this chart stacks
# THREE keys in one column and still clears the page's inner border at 0.018), but rows past the
# 15th are the least threatening shooters by construction and were padding the column rather
# than informing it. Their shots are still plotted under the "Other" rollup.
SHOT_LEGEND_MAX = 15

# Shot_Map.csv's `situation` field has 8 raw values, collapsed here to 6 marker shapes. Counts
# below are Columbus Crew's own 2026 MLS season (212 shots) — the three set-piece variants are
# merged because they're the same tactical situation to defend and individually too thin to
# read as distinct shapes (corner 31, set-piece 4, throw-in-set-piece 1).
#
# Penalties stay their OWN shape rather than merging into the dead-ball group: they're excluded
# from the npxG/90 ranking (see generate_shot_map), so a distinct marker is what lets a reader
# see that the cluster sitting on the penalty spot isn't what earned a player their rank.
SHOT_SITUATION_GROUPS = {
    'assisted': 'Assisted',                 # 127
    'corner': 'Set Piece',                  #  31
    'set-piece': 'Set Piece',               #   4
    'throw-in-set-piece': 'Set Piece',      #   1
    'regular': 'Open Play',                 #  27
    'fast-break': 'Fast Break',             #  13
    'free-kick': 'Free Kick',               #   5
    'penalty': 'Penalty',                   #   4
}

# Group -> marker, in legend display order. All are FILLED markers, so they honour the
# edgecolors/linewidths the sibling scatter charts in this file already use; an unfilled marker
# ('x', '+', '1') silently ignores both and would render as a bare stroke in the team colour.
# NOTE there is no 'Penalty' entry: penalties are dropped from this chart entirely (see
# SHOT_EXCLUDED_SITUATIONS), so giving them a shape would promise marks that are never drawn.
SHOT_SITUATION_MARKERS = {
    'Assisted': 'o',
    'Open Play': 's',
    'Fast Break': 'D',
    'Set Piece': '^',
    'Free Kick': 'v',
}

# Situations left off the shot map altogether. Every penalty is recorded at exactly the same
# point — (88.5, 50.0) after the frame conversion, the penalty spot — at a flat 0.7884 xG, so a
# season's worth stack the chart's largest markers on the single most crowded pixel it has, in
# the middle of the busiest region. They were already excluded from the xG heatmap and from the
# npxG/90 ranking for the same reason (a spot kick says nothing about open-play threat), so
# dropping the marks too makes the chart consistent rather than losing anything it was using.
#
# The cost is that the report no longer shows how many penalties a team wins. That is a
# different question from how they create chances, and this chart is not where it belongs.
SHOT_EXCLUDED_SITUATIONS = {'Penalty'}

# Window the shot map is cropped to. Not the halfway line — shots do not spread over the whole
# attacking half, they pack into the top of the box, and the wasted margin was costing
# resolution exactly where the marks are densest.
#
# Chosen off the real distribution rather than by eye. Across three squads (711 shots), the
# 0.5th percentile shot sits at x=69.3 and the 0.5th/99.5th at y=24.6/72.6, so this window drops
# 2 of 711 shots — 0.28% — and buys 1.40x linear magnification, near double the area. The
# discarded handful are 35-yard efforts that carry almost no xG between them.
#
# The window still contains the whole penalty area (opta y=21.1-78.9, x>=83) with margin, so the
# box lines a reader navigates by are all present.
SHOT_MAP_X_MIN = 65
SHOT_MAP_Y_MIN = 15
SHOT_MAP_Y_MAX = 85

# Heatmap grids for the shot and corner maps. bin_statistic always spans the FULL pitch, so these
# are chosen for what survives each chart's crop, and both were cut back hard from an earlier
# (20, 10) that was finer than the data could support. Measured on Nashville:
#
#   shot map    (20,10) -> 56 visible cells,  4.0 shots each,      29% empty
#               (12, 6) -> 16 visible cells, 14.1 shots each,       6% empty
#   corner map  (20,10) -> 80 visible cells,  1.4 deliveries each, 59% empty
#               ( 8, 6) -> 18 visible cells,  6.2 deliveries each, 22% empty
#
# At the old settings most corner cells were empty and the rest held a single delivery, so the
# colour was reporting which cell happened to catch a ball rather than where the danger is.
SHOT_HEATMAP_BINS = (12, 6)
CORNER_HEATMAP_BINS = (8, 6)

# Where the shot maps' key column starts, in the LEFT chart's axes fractions. At bbox width 0.26
# from x=0.03 this is figure x=0.408 — a nudge right of the old 1.40 (0.394), which sat only
# 0.021 clear of the colourbar's rotated label ending at 0.373.
#
# It cannot move much further. The column is boxed in by the conceded chart at x=0.600, and the
# tables are set in a monospace face that is wider than the proportional lists they replaced, so
# the widest player row already runs close to that edge. If a squad's longest surname ever pushes
# it over, the lever is the Player column width in situation_columns/player_columns, not this.
SHOT_LEGEND_ANCHOR_X = 1.42

# Left edge of every paired chart on the right half of a page (def key pass, def shot, def
# corner) — constant across all three pairs. Every key in this file hangs off a LEFT-hand chart
# and is auto-centred by stacked_legend() in the empty column between that chart's own right edge
# and this x, so the value has to be a constant rather than measured: the sibling axes usually
# doesn't exist yet when a left chart's own keys are being placed.
RIGHT_CHART_X0 = 0.60

# Vertical gap, in axes fractions, between one of the shot map's stacked keys and the next.
# Big enough to read as separate blocks rather than one run-on list, small enough to keep the
# stack from running off the page. The gap compounds down the column — the size key sits two
# gaps below the player key — so tightening it pulls the lower two keys up alongside the pitch
# instead of stranding them in the empty page below it.
SHOT_LEGEND_GAP = 0.05

# Bottom edge of the shot map's notes box. The box grows UPWARD from here to 0.03 below the
# pitch, so it absorbs whatever room the chart leaves rather than being a fixed height like the
# siblings'. 0.05 clears the page's inner border at 0.018.
SHOT_NOTES_Y0 = 0.05

# Marker AREA for a 0.0 xG shot and for a 1.0 xG one; everything between is linear in xG.
#
# A FIXED scale, not one normalised to each team's own biggest chance — the same trap the
# *_DIFF_SCALE constants above exist to avoid. Normalising per team would make a side whose best
# chance all season was 0.35 xG render that shot exactly as large as another team's tap-in, so
# marker size would mean something different in every opponent's report.
#
# Size rather than opacity is what carries xG, which was tested rather than assumed. Charlotte
# FC's xG distribution is heavily skewed — 59% of shots under 0.10, median 0.073 against a max
# of 0.917 — so a linear OPACITY map renders the median shot at 8% and the pitch comes out
# nearly blank; even a sqrt-scaled version washes the player colours out to pastel and breaks
# the colour-is-player key. Opacity also accumulates where marks overlap (the median shot
# overlaps 2 others, 22% overlap 3+), so a cluster of junk chances would read as dark as one
# genuine one. Area has neither problem.
#
# The top of the range is deliberately modest. Size and density compound on this chart — a high
# xG shot is by definition a central, close-range one, so the biggest markers land precisely
# where the marks are already most crowded. Anything larger closes up the front of the six-yard
# box into a solid mass.
#
# The FLOOR is raised well above where a linear scale would put it, because the distribution is
# skewed toward it: 59% of Charlotte's shots sit under 0.10 xG, and at a floor of 6 the majority
# of every chart rendered as specks too small to read a colour or a shape off — the two other
# encodings on the mark. The cost is compression at the bottom (0.00 vs 0.10 xG is now a subtler
# difference), which is the right trade: the low end is where xG differences matter least, and
# the top of the range is untouched so the genuine chances still stand out by size.
SHOT_SIZE_MIN = 16
SHOT_SIZE_MAX = 58

# The xG every mark is drawn at when the competition carries none (see has_xg). 0.20 rather than
# the midpoint: the real distribution is heavily skewed toward zero, so a mid-scale mark would
# make a no-xG chart read as far more dangerous than the same team's league chart. This lands the
# uniform mark just above where the bulk of real shots sit.
SHOT_UNIFORM_XG = 0.20

# Per-shape area multipliers. Matplotlib's `s` is marker AREA, which is not the same as apparent
# size across shapes — a star's points enclose a lot of empty space, so at equal area it reads
# noticeably smaller than a circle. These correct the ink, NOT the data: they're deliberately
# mild, because size now carries xG and a large shape factor would double-count (a penalty is
# already drawn big by its 0.7884 xG).
SHOT_SHAPE_SIZE_FACTORS = {
    'Set Piece': 1.2,   # '^'
    'Free Kick': 1.2,   # 'v'
}

# Fixed area for the shapes shown in the situation and delivery keys, where the marker stands for
# a category rather than for a particular shot's xG.
#
# 28 rather than 40, and the two numbers have to move together with KEY_TABLE_FONTSIZE: a legend
# row is as tall as the larger of its text and its handle, so once the tables dropped to 5.2pt the
# row pitch became ~6.8pt while these markers stayed ~6.3-6.9pt across, and consecutive shapes
# collided. 28 puts a marker at ~5.3pt inside the 8.3pt pitch KEY_TABLE_MARKER_SPACING gives it.
SHOT_LEGEND_MARKER_AREA = 28

# --- Corner maps (page four) -------------------------------------------------------------
# Takers are far more concentrated than shooters or key passers, so this cap is generous rather
# than binding: measured on Nashville, 8 players took a corner all season and the top TWO took
# 77% of them. The cap exists for the squad that spreads them around, not the normal case.
CORNER_TAKER_LEGEND_MAX = 8

# Both corner text lists. 10 rows is about where a reader stops scanning a name column, and on
# real data the tail past it is players with one or two events.
CORNER_LIST_MAX = 10

# The corner maps get their OWN window, wider than the shot maps'. The shot window's y=15-85 crop
# is right for shots, which pack into the middle, but wrong for corners in two ways: it cut 26 of
# 219 real deliveries (11.9%) — every short corner and every ball worked back to the touchline,
# which is exactly the routine a defence needs to see — and it cropped out the corner ARCS
# themselves, at y~0.5 and y~99.5, so the deliveries fanned in from off-screen with no visible
# origin. Corners therefore keep the FULL pitch width and only the length axis is cropped.
#
# x=60 clips nothing (the deepest delivery measured across both directions ends at x=71.2) while
# still dropping the 60% of the pitch a corner never reaches. It also sets the charts' shape: at
# 40 units long by 100 wide the visible pitch is 42m x 68m, so an aspect-locked bbox 0.26 of the
# page tall comes out 0.208 wide — narrower than the shot map's square, which is what buys the
# key column its room between the two charts.
CORNER_MAP_X_MIN = 60

# Marker by which corner arc the delivery was struck from, keyed on "is it the arc drawn at the
# BOTTOM of this chart". Deliberately as-drawn rather than left/right: the two teams' frames are
# related by a rotation (see rotate_into_opponent_frame), so an arc's numeric Y flips between the
# taken and conceded charts, and only a drawn-position label means the same thing on both. It is
# also the honest label — which physical touchline Y~0 is has not been established.
CORNER_ARC_MARKERS = {True: 'o', False: '^'}
CORNER_ARC_LABELS = {True: 'Bottom arc', False: 'Top arc'}

# One size for every delivery. Size is unused on these charts — xG created is carried by the
# heatmap, not by the marks — so a fixed area keeps the marks reading purely as position, shape
# and fill.
CORNER_MARKER_SIZE = 42

# A cleared delivery is drawn hollow, so its outline is the only thing carrying the taker's
# colour and has to be heavier than the hairline a filled marker uses for separation.
CORNER_HOLLOW_EDGE_WIDTH = 1.0
CORNER_MARKER_ALPHA = 0.85

# Bottom edge of page four's top-row notes boxes. The box grows UPWARD from here to 0.03 below
# the pitch, the same way the shot map's does and for the same reason — these bboxes are sized to
# the cropped pitch exactly, so there is no aspect-lock slack to derive an offset from. At the
# top row's bbox[1]=0.525 this yields the 0.12-high box the rest of the report uses.
CORNER_NOTES_Y0 = 0.375

# How a goal is set apart. Colour, shape and size are all spoken for (player, situation, xG), so
# what is left is the marker's outline or its fill pattern.
#
#   'bold'   - a heavier outline. The default, because it adds its ink on the PERIMETER, which
#              is also what separates one overlapping mark from another. In the crowded front of
#              the box that makes the chart easier to read rather than harder.
#   'stripe' - diagonal stripes through the marker (see striped_marker). More distinctive in
#              isolation, but it fills the interior, competing with the fill colour that carries
#              player identity and adding ink precisely where marks already overlap.
#   'both'   - stripes and a heavy outline together.
#
# Fill opacity is the same for goals as for everything else either way; only the outline changes.
SHOT_HEATMAP_ALPHA = 0.65

SHOT_GOAL_STYLE = 'bold'
SHOT_MARKER_EDGE_WIDTH = 0.3
SHOT_GOAL_EDGE_WIDTH = 1.0
SHOT_MARKER_ALPHA = 0.55

# Marker opacity for the CONCEDED shot map, which drops below SHOT_MARKER_ALPHA for exactly the
# reason generate_def_key_pass_map already drops 0.8 -> 0.45 against its own offensive sibling:
# the shot map's marks are spread across a tab20 palette and read as distinct even when they
# overlap, but these are all one colour, so at 0.55 a few hundred of them packed into the box
# merge into a solid slab that swamps the heatmap underneath — which is the layer carrying the
# xG magnitude. The lower opacity also makes overlapping marks legible as clusters, which is
# what a reader is looking for on a conceded map.
DEF_SHOT_MARKER_ALPHA = 0.40

# Spacing between stripes, in the units matplotlib's marker paths live in — every built-in
# marker fits inside [-0.5, 0.5], so 0.22 puts four or five stripes across a shape whatever its
# size on the page.
STRIPE_SPACING = 0.22

_striped_marker_cache = {}

def striped_marker(marker):
    '''`marker` with diagonal stripes built into the marker path itself.

    NOT matplotlib's `hatch`, which is what this replaces. A hatch passed to scatter applies to
    the whole PathCollection: matplotlib tiles one pattern in DEVICE space and clips it to each
    marker, so the stripes are anchored to the page rather than to the shape. Two effects, both
    wrong here — neighbouring markers show one continuous striped sheet rather than each having
    its own stripes, and because the tile size is fixed while the markers are not (size carries
    xG on this chart), a small marker catches a single stripe while a large one catches four.

    Baking the stripes into the path fixes both, because scatter scales the path by the marker
    size: every goal then shows the same stripes in the same places relative to its own shape,
    at every size and on every shape.

    The stripes are real segments clipped to the outline, computed by intersecting each stripe
    line with the marker's own polygon. They're appended as open MOVETO/LINETO subpaths, which
    enclose no area — so the fill still comes from the shape alone, while the edge stroke draws
    the outline and the stripes together in one colour.

    Returns (path, area_scale). MULTIPLY `s` BY area_scale or the striped marker comes out the
    wrong size: matplotlib renormalises any custom Path marker to fit [-0.5, 0.5], while several
    built-in markers are not drawn at that extent — 'D' is a unit square rotated 45 degrees and
    so spans 1.414 — so handing the same path back as a custom marker silently shrinks a striped
    diamond to 0.707 of its plain twin. area_scale undoes exactly that, and is 1.0 for the
    shapes already drawn at [-0.5, 0.5].'''
    if marker in _striped_marker_cache:
        return _striped_marker_cache[marker]

    style = MarkerStyle(marker)
    base = style.get_path().transformed(style.get_transform())

    # to_polygons flattens curves, so a circle becomes a fine polygon and the same
    # segment-intersection maths works for every shape.
    polygons = [np.asarray(polygon) for polygon in base.to_polygons()]

    vertices = list(base.vertices)
    codes = list(base.codes) if base.codes is not None else \
        [MarkerPath.MOVETO] + [MarkerPath.LINETO] * (len(base.vertices) - 1)

    if polygons:
        # Stripes run along `direction`; `normal` is the axis their spacing is measured on.
        direction = np.array([1.0, 1.0]) / np.sqrt(2)
        normal = np.array([1.0, -1.0]) / np.sqrt(2)
        projections = [point @ normal for polygon in polygons for point in polygon]

        offset = np.ceil(min(projections) / STRIPE_SPACING) * STRIPE_SPACING
        while offset < max(projections):
            # Where this stripe's line crosses the outline, as distances along `direction`.
            crossings = []
            for polygon in polygons:
                for start, end in zip(polygon[:-1], polygon[1:]):
                    start_side, end_side = start @ normal - offset, end @ normal - offset
                    if start_side == end_side:
                        continue
                    if (start_side <= 0 <= end_side) or (end_side <= 0 <= start_side):
                        crossed = start + (start_side / (start_side - end_side)) * (end - start)
                        crossings.append(crossed @ direction)

            # Pairing sorted crossings walks inside/outside/inside across the shape, so
            # concave markers (the star) get stripes only where there is actually ink.
            crossings.sort()
            for first, second in zip(crossings[::2], crossings[1::2]):
                vertices += [direction * first + normal * offset,
                             direction * second + normal * offset]
                codes += [MarkerPath.MOVETO, MarkerPath.LINETO]
            offset += STRIPE_SPACING

    path = MarkerPath(np.array(vertices), codes)
    # Mirrors matplotlib's own normalisation of custom Path markers (scale 0.5 / max |vertex|),
    # inverted, and applied to AREA because that is what `s` means.
    area_scale = (np.max(np.abs(path.vertices)) / 0.5) ** 2

    _striped_marker_cache[marker] = (path, area_scale)
    return _striped_marker_cache[marker]

#%%
def crop_shot_map(ax, mirrored=False):
    '''Crop a drawn pitch to the shooting window. Called again after every pitch.draw, which
    resets the limits.

    The limit at the goal end is mplsoccer's own rather than a hardcoded 0 or 100: it pads past
    the goal line so the goal frame itself is drawn, and cropping to a flat 100 sliced it off —
    the sibling charts show that little rectangle and this one lost it.

    mirrored=True is generate_def_shot_map's window: that chart mirrors its shots into the
    scouted team's frame, so the goal in question sits at x=0 and the window is the reflection of
    this one. It's the LEFT limit that has to be left to mplsoccer there, for the same reason.
    The two windows are the same size either way (35 x-units by 70 y-units), which is what lets
    both charts keep the square bbox generate_shot_map's signature comment derives.'''
    if mirrored:
        ax.set_xlim(ax.get_xlim()[0], 100 - SHOT_MAP_X_MIN)
    else:
        ax.set_xlim(SHOT_MAP_X_MIN, ax.get_xlim()[1])
    ax.set_ylim(SHOT_MAP_Y_MIN, SHOT_MAP_Y_MAX)

def crop_corner_map(ax, mirrored=False):
    '''Crop a drawn pitch to the corner window. Called again after every pitch.draw, which resets
    the limits.

    Length axis ONLY — see CORNER_MAP_X_MIN for why corners keep the full width where the shot
    maps crop it. Y is left at whatever mplsoccer set, which also keeps the touchlines drawn
    rather than running the marks to a bare edge. The limit at the goal end is mplsoccer's own,
    padded past the goal line so the goal frame renders, exactly as crop_shot_map does.'''
    if mirrored:
        ax.set_xlim(ax.get_xlim()[0], 100 - CORNER_MAP_X_MIN)
    else:
        ax.set_xlim(CORNER_MAP_X_MIN, ax.get_xlim()[1])

def goal_marker(marker):
    '''(shape, area_scale) for a shot that went in, per SHOT_GOAL_STYLE. Plain shapes need no
    area correction, so they pass 1.0 — see striped_marker for why the striped ones don't.'''
    if SHOT_GOAL_STYLE in ('stripe', 'both'):
        return striped_marker(marker)
    return marker, 1.0

def goal_edge_width():
    '''Outline weight for a goal. Only the 'stripe' style leaves it at the common weight: there
    the stripes already carry the signal, and thickening the outline on top of them buried the
    fill colour that identifies the player.'''
    return SHOT_MARKER_EDGE_WIDTH if SHOT_GOAL_STYLE == 'stripe' else SHOT_GOAL_EDGE_WIDTH

# xG values shown in the marker-size key. Chosen to span the real range a reader will meet:
# roughly the 25th percentile shot, a good chance, and a penalty.
SHOT_SIZE_LEGEND_VALUES = [0.05, 0.30, 0.75]


def shot_marker_area(xg, group):
    '''Marker area for one shot: linear in xG on a fixed 0-1 scale, times a mild per-shape
    correction. xG is clipped rather than scaled so a freak >1 value can't blow up the mark.'''
    scaled = SHOT_SIZE_MIN + (SHOT_SIZE_MAX - SHOT_SIZE_MIN) * np.clip(xg, 0, 1)
    return scaled * SHOT_SHAPE_SIZE_FACTORS.get(group, 1.0)


def legend_bottom_in_axes(page, ax, legend, fallback):
    '''Bottom edge of an already-placed legend, in `ax`'s axes fractions — where the next key in
    a stacked column hangs from.

    Measured rather than hardcoded because every key's height varies with the data: the shot
    map's player key swings between 13 and 21 rows depending on the squad, and its situation key
    drops rows for situations a team never used. A draw is needed first — legends have no window
    extent until something has laid them out. Fallbacks should be sized for a full-height key, so
    a failure errs toward too much gap rather than an overlap.'''
    try:
        page.canvas.draw()
        return ax.transAxes.inverted().transform((0, legend.get_window_extent().y0))[1]
    except Exception:
        return fallback

# Keys are rendered as aligned TABLES rather than as "Name (1.23, 4, +5.67)" lists: the numbers
# are what a reader scans down, and in a list they sit at a different x on every row because the
# name in front of them is a different length. A column only aligns in a MONOSPACE face —
# proportional fonts render a space narrower than a digit, so padding cannot square them up —
# which is why these two constants exist rather than the report's usual font.
#
# 5.2pt rather than the 6pt the lists used, because monospace is the wider face at equal size and
# the key column's width is the binding constraint on both pages. Page three is the tight one:
# its column has to fit between the shot map's colourbar label ending at x=0.408 and the conceded
# chart's pitch at x=0.600, a span of 0.192, and a 32-column table at 5.2pt runs about 0.164 of
# page width plus the swatch. At 5.5 it did not fit. See SHOT_LEGEND_ANCHOR_X.
KEY_TABLE_FONT = 'DejaVu Sans Mono'
KEY_TABLE_FONTSIZE = 5.2

# Row spacing for tables whose handles are MARKERS rather than colour swatches. A swatch is a
# small rectangle that never exceeds the text height, so those tables are fine at the 0.3 the
# other keys use; a marker is round and roughly as tall as the font, so at 0.3 the shapes in the
# situation and delivery keys ran into each other. 0.6 gives a 5.2 * 1.6 = 8.3pt row pitch against
# a ~5.3pt marker (see SHOT_LEGEND_MARKER_AREA).
KEY_TABLE_MARKER_SPACING = 0.6

def format_key_table(columns, rows):
    '''(header_string, [row_strings]) for a key drawn as a table.

    `columns` is [(header, width, align)] and `rows` is a list of value tuples in the same order.
    Everything is padded to a fixed character width, which only reads as a table in the monospace
    face these are drawn in — see KEY_TABLE_FONT.

    The header comes back as its own string rather than being folded into the legend title,
    because a title is laid out from the legend's left edge while row labels start after the
    handle. Rendering the header as a ROW with an invisible handle is what makes its columns land
    over the values beneath them, at any handle width.'''
    def pad(value, width, align):
        text = str(value)
        # Truncation would silently rename a player, so an over-long value pushes the column out
        # instead. The name widths below are set from the longest real surname for that reason.
        if align == '<':
            return text.ljust(width)
        if align == '^':
            return text.center(width)
        return text.rjust(width)

    header = ''.join(pad(head, width, align) for head, width, align in columns)
    body = [''.join(pad(value, width, align) for value, (_, width, align) in zip(row, columns))
            for row in rows]
    return header, body

def table_legend(ax, header, rows, title, anchor_x, top, handles=None, grey=None, **kwargs):
    '''A stacked key whose rows are an aligned table with a column header.

    `handles` supplies one artist per row (colour swatches, situation markers); omit it for a
    text-only key. `header` may be one string or several, for a table whose columns come in
    groups — the situation key pairs a shot count with a percentile twice over, and a single
    header row would label both pairs "Sh". Each header line gets an invisible handle so its
    columns line up with the values rather than with the legend's left edge.

    `grey` is an optional per-row bool marking rows to de-emphasise — same device the shot map's
    player key already uses for players short of the minutes threshold, applied to the text so the
    swatch keeps tying the row to its marks.'''
    header_lines = [header] if isinstance(header, str) else list(header)
    # The title is rendered as an extra row, centred with the same str.center() padding the
    # columns themselves use, rather than passed to Legend's own `title=`. Legend's built-in title
    # centres over the WHOLE row — handle column plus text — but every row here starts with an
    # invisible handle that reserves the same width a swatch would take, which the title has no
    # equivalent of. Centred that way, the title lands visibly left of the table a reader actually
    # sees. Centring the title text itself, in the same monospace column, keeps it aligned with
    # that table instead. Bold rather than a larger size, so its characters keep the exact advance
    # width format_key_table already padded the columns to.
    title_line = title.center(len(header_lines[0]))
    all_lines = [title_line] + header_lines
    all_handles = ([Line2D([], [], linestyle='none', marker='') for _ in all_lines] +
                   (list(handles) if handles is not None else
                    [Line2D([], [], linestyle='none', marker='') for _ in rows]))
    legend = stacked_legend(ax, all_handles, None, anchor_x, top,
                            labels=all_lines + list(rows),
                            prop={'family': KEY_TABLE_FONT, 'size': KEY_TABLE_FONTSIZE},
                            **kwargs)
    texts = legend.get_texts()
    texts[0].set_fontweight('bold')
    # Headers are column labels, not data — greyed so they read as furniture.
    for text in texts[1:1 + len(header_lines)]:
        text.set_color('#6b6b6b')
    for text, is_grey in zip(texts[1 + len(header_lines):], grey or []):
        if is_grey:
            text.set_color('#6b6b6b')
    return legend

def player_qualifier(matches, expected_lineup=None, expected_subs=None, minutes=None):
    '''Returns `qualified(player) -> bool`: is this player worth reading at full weight?

    Every player key on the report greys its unqualified rows, so the rule lives here rather than
    being restated at each one — the key-pass, shot, corner-taker, corner-shooter and
    corner-defender keys all call this.

    Greying answers one question: is this player expected on the pitch? The lineup prediction
    answers it directly, so it wins outright — a predicted starter stays black however few minutes
    they have banked. That matters: checked against the real 8-16-2026 fixture, Jamal Thiaré (21%
    of minutes) and Mohamed Farsi (26%) are both predicted to START, and a minutes-only rule greyed
    out two of the eleven players the coach most needs to plan for.

    Minutes and roster remain the fallback for everyone the model didn't name. The `or` also
    handles an unavailable prediction for free — when the predictor returns nothing (it does
    exactly this for a date with no scheduled fixture, giving XI=0/subs=0) the first term is simply
    always False and the rule collapses back to minutes-and-roster rather than greying the squad.

    The two prediction frames name the column differently — expected_lineup uses 'Player Name',
    expected_subs uses 'Player' — confirmed against live output. Every predicted sub sat at
    0.47-0.57 Sub In Probability, so no probability floor is applied: being in the predicted squad
    at all is the signal.'''
    predicted_to_feature = set()
    for frame, column in ((expected_lineup, 'Player Name'), (expected_subs, 'Player')):
        if frame is not None and len(frame) and column in frame.columns:
            predicted_to_feature |= set(frame[column].dropna())

    player_minutes = compute_player_minutes(matches) if minutes is None else minutes
    team_minutes = compute_total_minutes(matches)
    # None means "couldn't check", which must not grey anybody — see load_current_squad.
    current_squad = load_current_squad(matches)

    def qualified(player):
        played = player_minutes.get(player, 0)
        enough_minutes = team_minutes and (played / team_minutes) >= KEY_PASS_MINUTES_QUALIFIED
        on_roster = current_squad is None or player in current_squad
        return (player in predicted_to_feature) or bool(enough_minutes and on_roster)

    return qualified

def stacked_legend(ax, handles, title, anchor_x, top, labels=None, **kwargs):
    '''One block in a stacked key column, constructed and attached in a single consistent way.

    Every key here goes through Legend + add_artist rather than ax.legend(). ax.legend() does two
    things at once — it builds the legend AND installs it as the axes' single legend — so a
    column built with it is only correct by accident of ordering: each key has to be add_artist'd
    to survive the next ax.legend() call, and whichever key happens to be LAST must then skip the
    add_artist or it ends up registered twice and drawn on top of itself. That invariant breaks
    silently the moment a key is added, removed or reordered, which is exactly what happened when
    the corner charts' shooter list became conditional. Constructing the Legend directly leaves
    ax.legend_ untouched, so every block behaves the same however many there are.'''
    # fontsize and prop are alternative ways to say the same thing and Legend ignores the former
    # when the latter is given, so only pass fontsize when the caller hasn't supplied a font.
    if 'prop' not in kwargs:
        kwargs.setdefault('fontsize', 6)
    kwargs.setdefault('labelspacing', 0.3)
    legend = Legend(ax, handles,
                    labels if labels is not None else [handle.get_label() for handle in handles],
                    loc='upper left', bbox_to_anchor=(anchor_x, top), bbox_transform=ax.transAxes,
                    title=title, title_fontsize=7, frameon=False, **kwargs)
    ax.add_artist(legend)
    center_legend_in_gap(ax, legend, anchor_x, top)
    return legend

def _companion_colorbar_right_edge(ax):
    '''Figure-x of the right edge of this axes' own colorbar's label, if it has one — the space a
    key must clear before the empty column actually starts.

    Found by position rather than passed in, because every left-hand chart's colorbar is already
    on the page by the time its keys are placed (every caller draws its heatmap and colorbar
    before touching table_legend), so a narrow axes hugging this one's right edge over the same
    rows is exactly what `page.add_axes([pitch_pos.x1 + 0.02, pitch_pos.y0, 0.015,
    pitch_pos.height])` creates. Falls back to this axes' own right edge when nothing matches —
    the xG-less shot map skips its colorbar entirely, and the gap genuinely starts at the pitch.'''
    ax_pos = ax.get_position()
    for other in ax.figure.axes:
        if other is ax:
            continue
        other_pos = other.get_position()
        if (0 < other_pos.x0 - ax_pos.x1 < 0.05 and other_pos.width < 0.05
                and abs(other_pos.y0 - ax_pos.y0) < 0.01):
            label = other.yaxis.label
            if label.get_text():
                return ax.figure.transFigure.inverted().transform(
                    (label.get_window_extent().x1, 0))[0]
            return other_pos.x1
    return ax_pos.x1

def center_legend_in_gap(ax, legend, anchor_x, top):
    '''Re-anchor a just-built legend so its box is horizontally centred in the empty column
    between the inside edge of this chart (its colorbar and label, if it has one — see
    _companion_colorbar_right_edge — else the pitch itself) and RIGHT_CHART_X0, the sibling
    chart's left edge, rather than sitting wherever the caller's anchor_x happened to land it.

    A real draw is needed first to know the box's rendered width, the same requirement
    legend_bottom_in_axes has and for the same reason: nothing has a window extent until
    something has laid it out. The shift is then applied to anchor_x rather than computed as an
    absolute position, because bbox_to_anchor's transform (ax.transAxes) is an affine map, so a
    delta in axes fraction translates the box by that same delta in figure fraction regardless of
    where anchor_x started.'''
    ax.figure.canvas.draw()
    gap_left = _companion_colorbar_right_edge(ax)
    box = legend.get_window_extent()
    (box_x0, _), (box_x1, _) = ax.figure.transFigure.inverted().transform([(box.x0, 0), (box.x1, 0)])
    ax_pos = ax.get_position()
    gap_center = (gap_left + RIGHT_CHART_X0) / 2
    shift_axes = (gap_center - (box_x0 + box_x1) / 2) / ax_pos.width
    legend.set_bbox_to_anchor((anchor_x + shift_axes, top), transform=ax.transAxes)

def plot_corner_marks(ax, corners, color, alpha=CORNER_MARKER_ALPHA):
    '''Draw corner deliveries at the point they ENDED: shape = which arc as drawn, hollow = the
    defence cleared it.

    Split the same way plot_shot_marks splits, and shared by both corner charts so the taken and
    conceded maps cannot drift on what a shape or a hollow mark means — the conceded chart
    carries no key of its own and is decoded entirely by the key beside its sibling.

    `Arc` must already be expressed relative to THIS chart's drawn orientation; see
    generate_def_corner_map, which re-derives it after rotating.'''
    for arc, marker in CORNER_ARC_MARKERS.items():
        for completed in (True, False):
            subset = corners[(corners['Arc'] == arc) & (corners['Completed'] == completed)]
            if subset.empty:
                continue
            ax.scatter(subset['Pass End X Coord'], subset['Pass End Y Coord'],
                       marker=marker, s=CORNER_MARKER_SIZE,
                       facecolors=color if completed else 'none',
                       edgecolors='black' if completed else color,
                       linewidths=SHOT_MARKER_EDGE_WIDTH if completed else CORNER_HOLLOW_EDGE_WIDTH,
                       alpha=alpha,
                       # Deliveries that found a teammate ride above the cleared ones — the
                       # cluster a defence has to plan for shouldn't sit under the noise.
                       zorder=4 if completed else 3)

def has_xg(shots):
    '''Whether this shot frame carries xG at all.

    extract_shots guarantees the answer is all-or-nothing: it drops individual xG-less shots when
    the frame has some, and keeps every one when the frame has none. So a single check here tells
    a chart whether it can draw its xG layers — the heatmap, the marker sizing and the xG columns
    in its keys — or has to fall back to counts. See SHOT_XG_FALLBACK_NOTE for what that costs.'''
    return not shots.empty and bool(np.isfinite(shots['xg']).any())

# Printed into the affected charts' titles so a reader is never left to wonder why one opponent's
# report has a heatmap and the next one doesn't.
SHOT_XG_FALLBACK_NOTE = 'no xG in this competition'

def plot_shot_marks(ax, shot_rows, color, alpha=SHOT_MARKER_ALPHA, uniform_size=False):
    '''Draw a set of shots as marks on `ax`, all in one colour.

    Split twice: once per situation, so each mark gets its shape, and again into goals and
    non-goals, since the marker path differs between them. Sizes come from shot_marker_area, so
    within a call the marks still vary — `s` accepts an array.

    Shared by generate_shot_map (called once per shooter, and once for the "Other" rollup) and
    generate_def_shot_map (called once for the lot, since that chart has no player dimension).
    It lives here rather than as a closure in the first of those so the two charts cannot drift
    apart on what a goal, a set-piece mark or an xG-scaled area looks like — the whole premise of
    the conceded chart is that a reader decodes its marks using the other chart's keys.

    uniform_size drops the xG encoding for competitions that carry no xG (see has_xg). Every mark
    is then drawn at the size a mid-range chance would get, so the chart still reads as a shot map
    rather than as a field of specks — and crucially the size key is removed alongside it, so
    nothing on the page claims the marks mean anything they don't.

    Nothing here is labelled. The shot map's keys are built from Patch swatches and proxy
    handles, because no single real mark on that chart stands for a whole category.'''
    for group, marker in SHOT_SITUATION_MARKERS.items():
        group_shots = shot_rows[shot_rows['Situation Group'] == group]
        if group_shots.empty:
            continue
        scored = group_shots['shotType'] == 'goal'
        for is_goal, subset in ((False, group_shots[~scored]), (True, group_shots[scored])):
            if subset.empty:
                continue
            shape, area_scale = goal_marker(marker) if is_goal else (marker, 1.0)
            area = (shot_marker_area(SHOT_UNIFORM_XG, group) if uniform_size
                    else shot_marker_area(subset['xg'].to_numpy(), group))
            ax.scatter(subset['Shot X Coord'], subset['Shot Y Coord'], color=color,
                       marker=shape,
                       s=area * area_scale,
                       edgecolors='black',
                       linewidths=goal_edge_width() if is_goal else SHOT_MARKER_EDGE_WIDTH,
                       alpha=alpha,
                       # Goals ride above the misses so a converted chance is never buried
                       # under the cluster of attempts around it.
                       zorder=4 if is_goal else 3)

# Shot outcomes that never put the ball on target, and so carry an xGOT of zero by definition.
# MLS and Liga MX record that zero explicitly; NCAA leaves the cell blank, so extract_shots fills
# it in to keep the two comparable. Measured against the archive, the split is clean: in NCAA
# every one of 1,956 misses and 1,504 of 1,506 blocks have a blank xGOT, while all 1,496 saves
# have a value.
SHOT_OFF_TARGET_TYPES = {'miss', 'block', 'post'}

# Any `situation` value not in SHOT_SITUATION_GROUPS falls back to this rather than being
# dropped — a shot with an unrecognised situation is still a shot, and SofaScore adding a new
# value should not silently remove events from the chart.
SHOT_SITUATION_DEFAULT = 'Open Play'

# How much wider each notes box is than the chart bbox it sits under, split evenly on
# both sides of the shared center (see notes_x0 at each call site).
NOTES_WIDTH_PAD = 0.04

# Every notes box on the report is this size. Page two already rendered 0.32 x 0.12 while pages
# three and four had drifted to 0.28, 0.26, 0.248 and 0.208 wide, because each chart derived its
# box from its own bbox — and the corner charts' bboxes are narrow (they are portrait; see
# CORNER_MAP_X_MIN), so their boxes came out narrowest of all. A fillable notes field should not
# change size depending on which chart it happens to sit under.
NOTES_WIDTH = 0.32

# Fixed colour-scale half-ranges for page two's four team-vs-league diff heatmaps: each
# is used as vmin=-SCALE, vmax=+SCALE. FIXED, not derived per team — every one of these
# charts previously pinned its scale to its own team's np.nanmax(np.abs(diff)), which
# made the same colour mean a different magnitude in every opponent's report (measured on
# the buildup chart, that per-team max swung 5.4x, from 0.0025 for Club Necaxa to 0.0137
# for Querétaro, with both teams' worst cell rendering identically saturated). A fixed
# scale is what lets a reader tell "close to league average" from "genuine outlier".
#
# Values are the p95 of |team − league| measured across every team, taken from MLS rather
# than Liga MX. The two leagues' spreads differ ~2x, but that is sampling noise, not
# tactical variance: MLS has 18 matches/team vs Liga MX's 3, noise scales as 1/sqrt(n),
# and sqrt(18/3) = 2.45 — which matches the observed p95 ratios (buildup 2.36x, press
# 2.67x) almost exactly. MLS is therefore the honest calibration; a thin-sample league
# will clip until its season accumulates, which is the correct reading of data too noisy
# to act on. Re-measure these if the underlying metric definitions change.
GK_PASS_DIFF_SCALE = 5.0        # % of a keeper's own passes (MLS p95 5.20)
DEF_ACTION_DIFF_SCALE = 1.5     # % of a team's own defensive actions (MLS p95 1.33)
BUILDUP_DIFF_SCALE = 0.002      # press value per opponent action (MLS p95 0.0022)
PRESSING_DIFF_SCALE = 0.002     # press threat per build-out action (MLS p95 0.0018)

#%%
def diff_label_color(value, scale, cmap):
    '''Black or white cell-label text, whichever contrasts with the fill underneath.

    The diff heatmaps are drawn on a fixed symmetric scale (vmin=-scale, vmax=+scale),
    and both ends of the diverging colormap are built from team colors — so for a club
    with a dark primary or secondary, a saturated cell renders near-black and hardcoded
    black label text disappears into it. That got much more common once the colour scale
    was fixed rather than per-team normalised (a fifth of cells clip on a thin-sample
    league), and a clipped cell is exactly the one worth reading.

    Rather than assume which end is dark, this asks the actual colormap what colour the
    cell got and switches on its Rec. 709 relative luminance, so it adapts to whatever
    team colors the report is being generated in. NaN cells are never painted (the
    "No Actions" case), so they keep the default black.'''

    if not np.isfinite(value):
        return 'black'
    normalized = np.clip((value + scale) / (2 * scale), 0, 1)
    red, green, blue = cmap(normalized)[:3]
    luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
    return 'white' if luminance < 0.5 else 'black'

#%%
def resolve_team_id(matches):
    '''SofaScore team ID of the side whose match folder `matches` was loaded from, or None if
    it can't be pinned down.

    The scouted team is the only side appearing in every one of its own fixtures, so
    intersecting each match's home/away IDs leaves exactly it. Name-independent, which is the
    point: the local folder name ('Columbus_Crew'), the API name ('Columbus Crew') and the CSV's
    homeTeam.name don't always agree, and a string comparison that silently fails picks the
    WRONG team rather than raising.

    Returns None when the intersection doesn't collapse to a single ID — which is the expected
    result for a one-match input (two teams both appear in "every" match) — so callers can fall
    back rather than acting on an arbitrary pick.'''
    team_ids = None
    for match_files in matches.values():
        attrs = match_files.get('Match_Attributes.csv')
        if attrs is None or 'homeTeam.id' not in attrs.columns:
            continue
        ids = {int(attrs['homeTeam.id'].iloc[0]), int(attrs['awayTeam.id'].iloc[0])}
        team_ids = ids if team_ids is None else (team_ids & ids)

    if not team_ids or len(team_ids) != 1:
        return None
    return team_ids.pop()

#%%
_current_squad_cache = {}

def load_current_squad(team_matches):
    """Set of player names on the scouted team's CURRENT roster, or None if it can't be
    determined.

    Uses SofaScore's squad endpoint rather than inferring from recent appearances, because
    the local inference gets it backwards in both directions — checked against real
    Columbus Crew data: Wessam Abou Ali had not featured in the last five matchday squads
    but IS still on the roster (injured, not gone), while Diego Rossi appeared in those
    same five and is NOT (departed). "Hasn't played lately" and "has left" are different
    facts and only the endpoint knows the second.

    The team ID is recovered from the cached Match_Attributes rather than taken as a
    parameter — see resolve_team_id.

    Returns None (rather than an empty set) when the team can't be resolved or the request
    fails, so callers can tell "nobody has left" apart from "we don't know" and skip the
    greying instead of marking the whole squad as departed.
    """
    team_id = resolve_team_id(team_matches)
    if team_id is None:
        return None

    if team_id in _current_squad_cache:
        return _current_squad_cache[team_id]

    import Pre_Match_Report  # deferred: Pre_Match_Report imports this module in turn
    squad_json = Pre_Match_Report.scrapeURLData(f'https://www.sofascore.com/api/v1/team/{team_id}/players')
    if not squad_json:
        return None

    squad = {entry['player']['name'] for entry in squad_json.get('players', []) if entry.get('player')}
    squad = squad or None
    _current_squad_cache[team_id] = squad
    return squad

#%%
# tab20's entries 14 and 15 are its two greys (#7f7f7f and #c7c7c7), which collide with the
# '#999999' swatch the capped legends use for their "Other (N players)" rollup — at legend
# swatch size a reader can't reliably tell the 8th-ranked player's row from the catch-all row
# sitting a few lines below it. Dropping both leaves 18 colours, comfortably above every cap in
# this file (KEY_PASS_LEGEND_MAX=15, SHOT_LEGEND_MAX=15).
TAB20_GREY_INDICES = {14, 15}

def distinct_player_colors():
    '''Per-player scatter colours, most distinguishable first.

    tab20 is built as TEN HUE PAIRS — a saturated and a pale version of each colour at adjacent
    indices — so walking it in order hands consecutive players two shades of the same hue (#1/#2
    dark and light blue, #3/#4 dark and light orange...) which at scatter-dot size are genuinely
    hard to tell apart. Taking the even indices first gives all ten saturated hues before any
    pale variant appears, so the highest-ranked players are maximally distinct and a pale shade
    only ever sits far down the list from its dark partner.'''
    palette = plt.get_cmap('tab20')
    saturated = [palette(i) for i in range(0, 20, 2) if i not in TAB20_GREY_INDICES]
    pale = [palette(i) for i in range(1, 20, 2) if i not in TAB20_GREY_INDICES]
    return saturated + pale

#%%
def abbreviate_player_name(name):
    '''"Sekou Tidiany Bangoura" -> "S. Bangoura".

    The key-pass legend's width is set by its single longest entry, so one long name
    pushes the whole legend into the chart beside it — measured: Columbus Crew's list ran
    to x=0.613 and overlapped the conceded map starting at 0.600, while Pumas' shorter
    names stopped at 0.587 and cleared it. Shortening bounds the width by the longest
    SURNAME rather than the longest full name.

    Mononyms ("Juninho") are returned untouched instead of becoming "J. Juninho". The
    pitch cards in generatePredictedLineup already display surnames only, so abbreviating
    here is consistent with the report's existing style.'''
    parts = str(name).split()
    if len(parts) < 2:
        return str(name)

    # Walk back from the end while the preceding token is a surname particle, so compound
    # surnames survive. Taking only the last token turned "Wessam Abou Ali" into "W. Ali",
    # which is simply the wrong name — his surname is "Abou Ali". Middle names must still
    # be dropped though ("Sekou Tidiany Bangoura" -> "S. Bangoura"), and nothing about the
    # capitalisation or position distinguishes "Abou" from "Tidiany", so a particle list is
    # the only thing that separates the two cases.
    surname_start = len(parts) - 1
    while surname_start > 1 and parts[surname_start - 1].lower() in SURNAME_PARTICLES:
        surname_start -= 1

    return f"{parts[0][0]}. {' '.join(parts[surname_start:])}"

#%%
def draw_notes_box(page, bbox):
    '''Empty bordered rectangle reserving space for a real fillable PDF field, added in a
    post-processing pass over the finished report (matplotlib alone can't create
    interactive form fields — see PDF_Form_Fields.py). Returns bbox unchanged so callers
    can register it for that pass.'''
    ax = page.add_axes(bbox)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color('#888888')
    return bbox

#%%
def compute_total_minutes(matches):

    def get_col(df, col, default=0):
        '''Safe column lookup: 0 if the column doesn't exist (e.g. injuryTime3/4,
        which are only present when extra time was played) or its value is NaN.'''
        series = df.get(col)
        if series is None:
            return default
        val = series.iloc[0]
        return val if pd.notna(val) else default

    total_minutes = 0

    for match, match_files in matches.items():
        match_attributes = match_files.get('Match_Attributes.csv')

        # Match_Attributes.csv is sometimes absent (confirmed gaps in this
        # dataset) — fall back to a flat 90 rather than crashing on the lookups
        # below, all of which assume a real DataFrame.
        if match_attributes is None:
            total_minutes += 90
            continue

        # Every finished match has both regulation halves regardless of whether
        # either team scored in them — homeScore.period1/2 are goal counts, not
        # "did this period happen" flags, so they aren't used to gate this.
        try:
            minutes = match_attributes['defaultPeriodLength'].iloc[0] * 2
        except:
            minutes = 90

        minutes += get_col(match_attributes, 'time.injuryTime1')
        minutes += get_col(match_attributes, 'time.injuryTime2')

        # homeScore.extra1/extra2 are NaN (not 0, not missing) when extra time
        # wasn't played, and bool(nan) is True in Python — pd.notna() avoids
        # treating every match as if it went to extra time.
        if pd.notna(match_attributes.get('homeScore.extra1', pd.Series([None])).iloc[0]):
            try:
                minutes += match_attributes['defaultOvertimeLength'].iloc[0]
            except:
                minutes += 15
            minutes += get_col(match_attributes, 'time.injuryTime3')

        if pd.notna(match_attributes.get('homeScore.extra2', pd.Series([None])).iloc[0]):
            try:
                minutes += match_attributes['defaultOvertimeLength'].iloc[0]
            except:
                minutes += 15
            minutes += get_col(match_attributes, 'time.injuryTime4')

        total_minutes += minutes

    return total_minutes

#%%
def compute_player_minutes(matches):
    '''Total minutes played per player, summed across `matches`. Player_Statistics.csv's
    'Player Name' column joins cleanly against Player_Event_Data.csv's 'Player' column
    (confirmed on real data — every event-data player has a matching stats row; the only
    one-sided names are unused substitutes with NaN minutes, who never appear in event
    data either).'''
    minutes = {}
    for match_files in matches.values():
        stats = match_files.get('Player_Statistics.csv')
        if stats is None:
            continue
        for _, row in stats.dropna(subset=['minutesPlayed']).iterrows():
            minutes[row['Player Name']] = minutes.get(row['Player Name'], 0) + row['minutesPlayed']
    return minutes

#%%
def get_files(current_path):
    '''Returns a list of all visible files inside the folder at currentPath'''
    return [f for f in os.listdir(current_path) if not f.startswith('.')]

#%%
# load_league_data re-reads every team's Player_Event_Data.csv off disk, and page two of
# the report now asks for a league baseline four times (GK passes, defensive actions,
# buildup, pressing). It's a pure function of its arguments, so the results are memoised
# here rather than paying that cost once per chart. Keyed on the full argument tuple
# since `data_required` and `exclude_team` both change what gets loaded.
_league_data_cache = {}

def load_league_data(leagues, date, data_required, exclude_team=None):
    '''`leagues` is a list (same convention as loadMatchData) — every team's
    matches across all listed leagues are pooled. `exclude_team` skips a single
    team folder (used to keep a team's own matches out of its own league-average
    baseline).

    Results are cached in-process (see _league_data_cache). Callers treat the return
    value as read-only — none of the chart functions mutate the returned frames in
    place, they filter/copy first.'''

    cache_key = (tuple(leagues), date, tuple(data_required), exclude_team)
    if cache_key in _league_data_cache:
        return _league_data_cache[cache_key]

    matches = {}
    seen_match_ids = set()
    year = date.split('-')[2]
    for league in leagues:
        league_path = f'/Users/jakeholfinger/Desktop/CC Analyst/Data/SofaScore_Data/{year}_Data/{league.replace(" ", "_")}_Data'
        if not os.path.exists(league_path):
            print(f'Data for {league} not found. Please scrape relevant data.')
            continue
        for team_folder in get_files(league_path):
            team_name = team_folder.removesuffix('_Data')
            if exclude_team is not None and team_name == exclude_team:
                continue
            team_path = os.path.join(league_path, team_folder)
            if not os.path.isdir(team_path):
                continue
            for match_folder in get_files(team_path):
                match_path = os.path.join(team_path, match_folder)
                if not os.path.isdir(match_path):
                    continue
                scraped_files = {}
                for file in get_files(match_path):
                    if file in data_required:
                        file_path = os.path.join(match_path, file)
                        try:
                            scraped_files[file] = pd.read_csv(file_path)
                        except Exception:
                            # Same tolerance loadMatchData has: a file that won't parse counts as
                            # absent rather than taking down a whole-league scan. 68 CSVs in the
                            # archive are zero-length and pd.read_csv raises on those.
                            continue

                # Match_Attributes.csv isn't always present (confirmed gaps in this
                # dataset) — read it if it exists and keep it in scraped_files (needed
                # downstream for per-90 minutes), skip the dedup/minutes info if not,
                # rather than crashing on a bare read.
                match_attributes_path = os.path.join(match_path, 'Match_Attributes.csv')
                match_attributes = None
                if os.path.exists(match_attributes_path):
                    match_attributes = pd.read_csv(match_attributes_path)
                    scraped_files['Match_Attributes.csv'] = match_attributes

                if match_attributes is not None and 'id' in match_attributes.columns:
                    match_id = match_attributes['id'].iloc[0]
                    if match_id in seen_match_ids:
                        print(f'Skipping duplicate match folder {match_folder} (match ID {match_id})')
                        continue
                    seen_match_ids.add(match_id)

                matches[match_folder] = scraped_files

    def folder_date(folder_name):
        return datetime.strptime(folder_name.split('_')[0], '%m-%d-%Y')

    sorted_matches = dict(sorted(matches.items(), key=lambda item: folder_date(item[0])))

    # Keep only matches that kicked off strictly before the target date.
    month, day, yr = date.split('-')
    target_date = datetime(int(yr), int(month), int(day)).date()
    filtered_matches = {
        folder: data for folder, data in sorted_matches.items()
        if folder_date(folder).date() < target_date
    }

    _league_data_cache[cache_key] = filtered_matches
    return filtered_matches

#%%
def load_actual_opponent_events(matches, leagues, date, team):
    '''`matches` (e.g. oppositionMatches) holds `team`'s own event data — each match's own
    Player_Event_Data.csv contains only `team`'s own events, never the actual team they
    played against that day (confirmed: a single match folder's file lists ~16-18 players,
    one full roster, not two). This finds, for each of `team`'s own fixtures, the real
    opponent's own event data for that SAME match — matched by SofaScore match ID rather
    than folder date, since the same match can appear under differently-dated folder names
    depending on scrape-time timezone (see loadMatchData/load_league_data).

    Returns a dict shaped like `matches` itself (folder name -> {filename: DataFrame}),
    just sourced from each real opponent's own team folder instead of `team`'s.'''
    target_ids = set()
    for match_files in matches.values():
        attrs = match_files.get('Match_Attributes.csv')
        if attrs is not None and 'id' in attrs.columns:
            target_ids.add(attrs['id'].iloc[0])

    if not target_ids:
        return {}

    # Pools every other team's matches across the given leagues, then keeps only the
    # folders whose match ID is one of team's own fixtures — i.e. exactly the real
    # opponents' own data for these specific matches, discarding everyone else pooled.
    league_matches = load_league_data(leagues, date, ['Player_Event_Data.csv'], exclude_team=team)

    opponent_matches = {}
    for match_folder, match_files in league_matches.items():
        attrs = match_files.get('Match_Attributes.csv')
        if attrs is not None and 'id' in attrs.columns and attrs['id'].iloc[0] in target_ids:
            opponent_matches[match_folder] = match_files

    return opponent_matches

#%%
def extract_gk_passes(matches):

    matches_gk_passes = []
    for match, match_files in matches.items():
        if 'Player_Event_Data.csv' not in match_files or 'Player_Statistics.csv' not in match_files:
            continue

        player_data = match_files['Player_Statistics.csv']
        gks = player_data[player_data['Position'] == 'G']['Player Name']

        event_data = match_files['Player_Event_Data.csv']
        match_gk_passes = event_data[(event_data['Event Type'] == 'pass') & (event_data['Player'].isin(gks))]
        matches_gk_passes.append(match_gk_passes)

    if not matches_gk_passes:
        return pd.DataFrame(columns=['Player', 'Event Type', 'Player X Coord', 'Player Y Coord'])

    gk_passes = pd.concat(matches_gk_passes)

    return gk_passes

# bbox[1]=0.4683 (this and its lower-row/def-action/pressing siblings) was solved, not
# guessed: measured the actual rendered pixel position of each chart's title (which sits
# well below the nominal bbox top due to aspect-lock shrink+centering) against a real
# page-two render, then solved for the y0 that makes three gaps equal — separator line to
# upper-row title, upper-row notes-box bottom to lower-row title, and lower-row notes-box
# bottom to the page's inner border. Changing bbox height/width invalidates this value.
def generate_relative_gk_pass_map(leagues, team, team_matches, date, team_colors, page, bbox=[0.13, 0.4683, 0.28, 0.42]):

    # extract_gk_passes needs Player_Statistics.csv to identify which player names
    # are goalkeepers — without it every league match fails that guard and the
    # league side of the comparison is silently always empty.
    league_matches = load_league_data(leagues, date, ['Player_Event_Data.csv', 'Player_Statistics.csv'], exclude_team=team)

    team_gk_passes = extract_gk_passes(team_matches)
    league_gk_passes = extract_gk_passes(league_matches)

    if team_gk_passes.empty or league_gk_passes.empty:
        print('No GK pass data available — skipping gk relative pass map.')
        return page, None

    def relative_vectors_yards(gk_passes):
        '''Pass End minus player touch location, in yards, per axis — origin (0,0)
        is where the keeper touched the ball, not a spot on the pitch. Converted to
        yards on each raw endpoint before differencing (linearly equivalent to
        scaling the difference, just matches this project's established
        opta-to-yards convention).'''
        start_x = gk_passes['Player X Coord'] / 100 * PITCH_LENGTH_YARDS
        start_y = gk_passes['Player Y Coord'] / 100 * PITCH_WIDTH_YARDS
        end_x = gk_passes['Pass End X Coord'] / 100 * PITCH_LENGTH_YARDS
        end_y = gk_passes['Pass End Y Coord'] / 100 * PITCH_WIDTH_YARDS
        dx, dy = end_x - start_x, end_y - start_y
        finite = np.isfinite(dx) & np.isfinite(dy)
        return dx[finite], dy[finite]

    team_dx, team_dy = relative_vectors_yards(team_gk_passes)
    league_dx, league_dy = relative_vectors_yards(league_gk_passes)

    # Relative pass vectors legitimately span negative values (backward/left of the
    # keeper's touch) — pitch.bin_statistic only bins points inside a real pitch's
    # [0,100] opta range and silently drops anything outside it, so this uses a
    # plain numpy histogram over a custom, off-pitch coordinate range instead.
    #
    # Extent is asymmetric (forward-biased), matching real keeper behavior — backward
    # passes rarely go more than ~20yd, forward punts can travel most of the pitch —
    # and its total span (115 x 74) matches PITCH_LENGTH_YARDS/PITCH_WIDTH_YARDS so
    # this panel renders at the same landscape aspect ratio as the real pitch in
    # generate_defensive_action_map, and its (6,5) bins are the same physical yards²
    # size as that map's, keeping per-90 counts comparable across the two panels.
    # Forward/backward is the x (horizontal) axis here to match that map's
    # "→ Attack →" orientation, rather than the old vertical layout.
    bins = (5, 4)
    x_extent = (-25, 100)
    y_extent = (-40, 40)
    x_edges = np.linspace(*x_extent, bins[0] + 1)
    y_edges = np.linspace(*y_extent, bins[1] + 1)

    team_gk_pass_grid, _, _ = np.histogram2d(team_dx, team_dy, bins=bins, range=[x_extent, y_extent])
    league_gk_pass_grid, _, _ = np.histogram2d(league_dx, league_dy, bins=bins, range=[x_extent, y_extent])

    # Convert both to each side's own share of their total GK passes (not a per-90 count)
    # before differencing — this isolates WHERE a keeper concentrates their passing from
    # HOW MUCH they pass overall, which per-90 counts conflate (a keeper who plays short
    # far less often than average would look uniformly "cold" everywhere in a per-90
    # diff, even if their actual distribution shape is completely normal).
    team_total = team_gk_pass_grid.sum()
    league_total = league_gk_pass_grid.sum()
    team_gk_pass_pct = team_gk_pass_grid / team_total * 100 if team_total else team_gk_pass_grid
    league_gk_pass_pct = league_gk_pass_grid / league_total * 100 if league_total else league_gk_pass_grid

    diff_gk_pass_grid = team_gk_pass_pct - league_gk_pass_pct

    # This is diverging data (more than league average vs. less, around a
    # meaningful zero) — team's primary color for "more passes than average" in a
    # zone, secondary color for "less," white at zero.
    cmap = LinearSegmentedColormap.from_list('diff_cmap', [team_colors[1], '#ffffff', team_colors[0]])

    ax = page.add_axes(bbox)
    ax.set_xlim(x_extent)
    ax.set_ylim(y_extent)
    ax.set_aspect('equal')
    for spine in ax.spines.values():
        spine.set_color('#888888')

    # Dashed crosshair through (0,0) — the keeper's own touch location — standing in
    # for the real pitch outline that generate_defensive_action_map draws (there's no
    # actual pitch to draw here since these are relative displacement vectors, not
    # absolute position).
    ax.axvline(0, color='#888888', linewidth=1, linestyle='--')
    ax.axhline(0, color='#888888', linewidth=1, linestyle='--')

    # histogram2d's grid is (nx, ny); pcolormesh wants (ny, nx).
    mesh = ax.pcolormesh(x_edges, y_edges, diff_gk_pass_grid.T, cmap=cmap,
                         vmin=-GK_PASS_DIFF_SCALE, vmax=GK_PASS_DIFF_SCALE)

    # Redraw the crosshair on top of the heatmap fill so it's visible — pcolormesh
    # defaults to zorder=1, same convention as generate_defensive_action_map's
    # pitch-line redraw (there the artists default to zorder=0.9 and need bumping;
    # here the crosshair is redrawn fresh on top for the same reason: to guarantee
    # it renders above the mesh regardless of default zorder ordering).
    existing_artists = set(ax.get_children())
    ax.axvline(0, color='#888888', linewidth=1, linestyle='--')
    ax.axhline(0, color='#888888', linewidth=1, linestyle='--')
    for artist in ax.get_children():
        if artist not in existing_artists:
            artist.set_zorder(mesh.get_zorder() + 1)

    # A dedicated colorbar axes, sized off ax's *actual* resolved position rather
    # than colorbar(ax=ax, ...)'s default "steal space from ax" behavior — the
    # latter shrinks ax a second time, which double-triggers the aspect enforcement
    # above and throws the colorbar/title out of alignment with the plot.
    pitch_pos = ax.get_position()
    cbar_ax = page.add_axes([pitch_pos.x1 + 0.02, pitch_pos.y0, 0.015, pitch_pos.height])
    # extend='both': the scale is now fixed (see the *_DIFF_SCALE constants), so a cell
    # can genuinely exceed it — arrow caps flag that instead of silently flattening an
    # extreme value into the same colour as an ordinary one.
    cbar = page.colorbar(mesh, cax=cbar_ax, extend='both')
    cbar.set_label('% of GK Passes (Team − League)', fontsize=8)

    x_centers = (x_edges[:-1] + x_edges[1:]) / 2
    y_centers = (y_edges[:-1] + y_edges[1:]) / 2
    for i, xc in enumerate(x_centers):
        for j, yc in enumerate(y_centers):
            ax.text(xc, yc, f'{diff_gk_pass_grid[i, j]:+.1f}%',
                    color=diff_label_color(diff_gk_pass_grid[i, j], GK_PASS_DIFF_SCALE, cmap), fontsize=7,
                     ha='center', va='center', zorder=mesh.get_zorder() + 2)

    ax.set_title('Relative GK Passes vs. League Average', fontsize=10, pad=12)
    ax.set_xlabel('Backwards ← Yards From Touch → Forward', labelpad=10, fontsize=8)
    ax.set_ylabel('Left ← Yards From Touch → Right', labelpad=10, fontsize=8)

    # bbox[1] is the requested box's bottom edge, not where the plot actually ends —
    # set_aspect('equal') shrinks and vertically centers the plot within that box. 0.06
    # is a fixed offset down from bbox[1] tuned to clear the tick labels and x-axis
    # label text below the plot (checked against a real render). generate_defensive_
    # action_map uses this same NOTES_OFFSET/bbox[1]=0.50 pairing so the two notes boxes
    # land on the same horizontal line. Centered on the left half of the page (x=0.25)
    # rather than under the chart's own bbox[0], which varies slightly chart to chart.
    # Anchored on the same TOP edge as before (old_y0 + old_height=0.06) and grown
    # downward from there — growing from y0 upward instead ate into the plot above it.
    notes_width = bbox[2] + NOTES_WIDTH_PAD
    notes_x0 = 0.25 - notes_width / 2
    notes_y0 = bbox[1] + NOTES_OFFSET - 0.06
    notes_bbox = draw_notes_box(page, [notes_x0, notes_y0, notes_width, 0.12])

    return page, notes_bbox
#%%
def extract_def_actions(matches):

    matches_def_actions = []
    for match, match_files in matches.items():
        if 'Player_Event_Data.csv' not in match_files:
            continue
        event_data = match_files['Player_Event_Data.csv']
        match_def_actions = event_data[event_data['Event Type'] == 'Defensive Action']
        matches_def_actions.append(match_def_actions)

    if not matches_def_actions:
        # Includes Event Sub-Type even though generate_defensive_action_map itself never
        # reads it — generate_pressing_turnover_map filters on it via this same
        # extractor, and would otherwise KeyError on this empty-data fallback path.
        return pd.DataFrame(columns=['Player', 'Event Type', 'Event Sub-Type', 'Player X Coord', 'Player Y Coord'])

    def_actions = pd.concat(matches_def_actions)

    return def_actions

#%%
def generate_defensive_action_map(leagues, team, team_matches, date, team_colors, page, bbox=[0.59, 0.4683, 0.28, 0.42]):

    league_matches = load_league_data(leagues, date, ['Player_Event_Data.csv'], exclude_team=team)

    # Get defensive action data
    team_def_actions = extract_def_actions(team_matches)
    league_def_actions = extract_def_actions(league_matches)

    if team_def_actions.empty or league_def_actions.empty:
        print('No defensive action data available — skipping defensive action map.')
        return page, None

    ax = page.add_axes(bbox)
    # pitch_color='none' so the pitch's own background fill doesn't paint over the
    # heatmap when the lines get redrawn on top of it below.
    pitch = Pitch(pitch_type='opta', pitch_color='none', line_color='#888888', linewidth=1)
    pitch.draw(ax=ax)

    team_def_action_bins = pitch.bin_statistic(team_def_actions['Player X Coord'], team_def_actions['Player Y Coord'], statistic='count', bins=(6, 5))
    league_def_action_bins = pitch.bin_statistic(league_def_actions['Player X Coord'], league_def_actions['Player Y Coord'], statistic='count', bins=(6, 5))

    # Convert both to each side's own share of their total defensive actions (not a
    # per-90 count) before differencing — this isolates WHERE a team concentrates its
    # defending from HOW MUCH defending it does overall, which per-90 counts conflate
    # (a low-possession team that defends more everywhere would look uniformly "hot"
    # in a per-90 diff, even if its actual shape is unremarkable).
    team_total = team_def_action_bins['statistic'].sum()
    league_total = league_def_action_bins['statistic'].sum()
    team_def_action_bins['statistic'] = team_def_action_bins['statistic'] / team_total * 100 if team_total else team_def_action_bins['statistic']
    league_def_action_bins['statistic'] = league_def_action_bins['statistic'] / league_total * 100 if league_total else league_def_action_bins['statistic']

    # bin_statistic returns a dict, not an array — diff the 'statistic' grids
    # specifically and carry the rest (grid edges, etc.) from the team's own bins.
    diff_def_action_bins = dict(team_def_action_bins)
    diff_def_action_bins['statistic'] = team_def_action_bins['statistic'] - league_def_action_bins['statistic']

    # This is diverging data (more than league average vs. less, around a
    # meaningful zero) — team's primary color for "more defensive actions than
    # average" in a zone, secondary color for "less," white at zero.
    cmap = LinearSegmentedColormap.from_list('diff_cmap', [team_colors[1], '#ffffff', team_colors[0]])

    mesh = pitch.heatmap(diff_def_action_bins, ax=ax, cmap=cmap,
                         vmin=-DEF_ACTION_DIFF_SCALE, vmax=DEF_ACTION_DIFF_SCALE)

    # Redraw the pitch lines on top of the heatmap fill so they're visible —
    # mplsoccer's line artists default to zorder=0.9, below pcolormesh's
    # zorder=1, so drawing them again isn't enough on its own; each new artist's
    # zorder has to be bumped above the mesh's or they stay hidden underneath it.
    existing_artists = set(ax.get_children())
    pitch.draw(ax=ax)
    for artist in ax.get_children():
        if artist not in existing_artists:
            artist.set_zorder(mesh.get_zorder() + 1)

    # A dedicated colorbar axes, sized off ax's *actual* resolved position rather
    # than colorbar(ax=ax, ...)'s default "steal space from ax" behavior — the
    # latter shrinks ax a second time, which double-triggers mplsoccer's aspect
    # enforcement and throws the colorbar/title out of alignment with the pitch.
    pitch_pos = ax.get_position()
    cbar_ax = page.add_axes([pitch_pos.x1 + 0.02, pitch_pos.y0, 0.015, pitch_pos.height])
    # extend='both': the scale is now fixed (see the *_DIFF_SCALE constants), so a cell
    # can genuinely exceed it — arrow caps flag that instead of silently flattening an
    # extreme value into the same colour as an ordinary one.
    cbar = page.colorbar(mesh, cax=cbar_ax, extend='both')
    cbar.set_label('% of Def Actions (Team − League)', labelpad=8, fontsize=8)

    # label_heatmap only accepts a single colour for every label, so the per-cell
    # contrast switch is applied afterwards to the artists it returns. They come back in
    # np.ravel order over the statistic grid (no cells are excluded — exclude_zeros /
    # exclude_nan both default to False), so they zip directly against the values.
    def_action_labels = pitch.label_heatmap(diff_def_action_bins, ax=ax, str_format='{:+.1f}%',
                                            color='black', fontsize=7, ha='center', va='center')
    for annotation, value in zip(def_action_labels, np.ravel(diff_def_action_bins['statistic'])):
        annotation.set_color(diff_label_color(value, DEF_ACTION_DIFF_SCALE, cmap))

    ax.set_title('Defensive Actions vs. League Average', fontsize=10, pad=12)
    #←
    ax.set_xlabel('→   Attack   →', labelpad=8, fontsize=8)

    # Shares NOTES_OFFSET and bbox[1]=0.50 with generate_relative_gk_pass_map so the two
    # notes boxes land on the same horizontal line (see that function for why bbox[1]
    # rather than the pitch's own resolved position is used as the anchor). Centered on
    # the right half of the page (x=0.75), mirroring the left-side charts. Anchored on
    # the same TOP edge as before (old_y0 + old_height=0.06) and grown downward from
    # there — growing from y0 upward instead ate into the plot above it.
    notes_width = bbox[2] + NOTES_WIDTH_PAD
    notes_x0 = 0.75 - notes_width / 2
    notes_y0 = bbox[1] + NOTES_OFFSET - 0.06
    notes_bbox = draw_notes_box(page, [notes_x0, notes_y0, notes_width, 0.12])

    return page, notes_bbox

#%%
# Karun Singh's publicly published Expected Threat (xT) grid — an 8 (width) x 12 (length)
# matrix of possession-value by zone, fit on 2017/18 Premier League event data. Column 0 =
# deep in a team's own defensive third, column 11 = edge of the opponent's box; values rise
# toward the attacking goal. Source: https://karun.in/blog/data/open_xt_12x8_v1.json — the
# exact public data file reused across the analytics community (e.g. cited by the
# socceraction package) rather than re-derived per-project, since this project's own event
# volume is too thin to fit a stable 96-zone model from scratch.
XT_GRID = np.array([
    [0.00638303, 0.00779616, 0.00844854, 0.00977659, 0.01126267, 0.01248344, 0.01473596, 0.0174506,  0.02122129, 0.02756312, 0.03485072, 0.0379259 ],
    [0.00750072, 0.00878589, 0.00942382, 0.0105949,  0.01214719, 0.0138454,  0.01611813, 0.01870347, 0.02401521, 0.02953272, 0.04066992, 0.04647721],
    [0.0088799,  0.00977745, 0.01001304, 0.01110462, 0.01269174, 0.01429128, 0.01685596, 0.01935132, 0.0241224,  0.02855202, 0.05491138, 0.06442595],
    [0.00941056, 0.01082722, 0.01016549, 0.01132376, 0.01262646, 0.01484598, 0.01689528, 0.0199707,  0.02385149, 0.03511326, 0.10805102, 0.25745362],
    [0.00941056, 0.01082722, 0.01016549, 0.01132376, 0.01262646, 0.01484598, 0.01689528, 0.0199707,  0.02385149, 0.03511326, 0.10805102, 0.25745362],
    [0.0088799,  0.00977745, 0.01001304, 0.01110462, 0.01269174, 0.01429128, 0.01685596, 0.01935132, 0.0241224,  0.02855202, 0.05491138, 0.06442595],
    [0.00750072, 0.00878589, 0.00942382, 0.0105949,  0.01214719, 0.0138454,  0.01611813, 0.01870347, 0.02401521, 0.02953272, 0.04066992, 0.04647721],
    [0.00638303, 0.00779616, 0.00844854, 0.00977659, 0.01126267, 0.01248344, 0.01473596, 0.0174506,  0.02122129, 0.02756312, 0.03485072, 0.0379259 ],
])

def lookup_xt(x, y, attacking_toward_100=True):
    '''xT for a team attacking toward x=100 (this file's opta-coordinate convention) at
    pitch position (x, y), each in 0-100. Pass attacking_toward_100=False for the
    opposite direction (mirrors x) — needed here because when the scouted team loses the
    ball, the opponent attacks toward the scouted team's OWN goal (x=0 in this file's own
    attacking-direction convention), not toward x=100.'''
    eval_x = x if attacking_toward_100 else (100 - x)
    col = np.clip((np.asarray(eval_x) / 100 * 12).astype(int), 0, 11)
    row = np.clip((np.asarray(y) / 100 * 8).astype(int), 0, 7)
    return XT_GRID[row, col]

#%%
def extract_offensive_turnover_actions(matches):

    matches_offensive_turnover_actions = []
    for match, match_files in matches.items():
        if 'Player_Event_Data.csv' not in match_files:
            continue
        event_data = match_files['Player_Event_Data.csv']
        # Crosses excluded: a blocked/cleared cross is a normal defensive contest in the
        # box, not a buildup mistake — including them concentrated a lot of "failures"
        # (crosses fail often but cheaply) in the attacking third, muddying the signal
        # this metric is meant to isolate (confirmed by inspecting the actual per-cell
        # numbers — that column's risk was coming from cross-failure rate, not genuine
        # buildup turnover cost).
        match_offensive_turnover_actions = event_data[event_data['Event Type'].isin(['pass', 'dribble'])]
        matches_offensive_turnover_actions.append(match_offensive_turnover_actions)

    if not matches_offensive_turnover_actions:
        return pd.DataFrame(columns=['Player', 'Event Type', 'Outcome', 'Player X Coord', 'Player Y Coord', 'Pass End X Coord', 'Pass End Y Coord'])

    turnover_actions = pd.concat(matches_offensive_turnover_actions)

    return turnover_actions

#%%
def buildup_grids(matches):
    '''Per-zone buildup risk/reward grids over the defensive half, for one pool of
    matches. Factored out of generate_buildup_turnover_map so the scouted team and the
    league-average baseline are computed by identical code — the chart differences the
    two. Returns the press_value / turnover_pct / total_actions grids plus bin edges.'''

    offensive_turnover_actions = extract_offensive_turnover_actions(matches)

    # Restrict to the team's own defensive half — the half they actually build out from.
    # (x=0 is a team's own goal in this file's coordinate convention, confirmed via
    # goalkeeper pass locations clustering around x=5-12.) The attacking half mixes in a
    # different phenomenon — congestion-forced turnovers near the opponent's box, driven
    # by a much higher failure rate on cheap-to-lose actions — that isn't about buildup
    # pressing at all and was muddying the signal this chart is meant to isolate.
    offensive_turnover_actions = offensive_turnover_actions[offensive_turnover_actions['Player X Coord'] <= 50]

    # Nothing to grid. Returned as None rather than as empty arrays because the arithmetic below
    # cannot survive an empty frame: its columns come back object-dtyped, and np.isfinite on an
    # object dtype raises "ufunc 'isfinite' not supported" — inside this function, before any
    # value the caller could have inspected. A competition SofaScore publishes no event data for
    # (NCAA) empties both the team pool and the league pool at once, so the caller must check both.
    if offensive_turnover_actions.empty:
        return None

    offensive_turnovers = offensive_turnover_actions[offensive_turnover_actions['Outcome'].isna()].copy()

    # A failed dribble is lost right where the dribbler was standing (Pass End X/Y Coord
    # is always NaN for dribbles — there's no separate end location). A failed pass/cross
    # is actually recovered by the opponent wherever it was headed, which can be a very
    # different zone than where it was struck from (e.g. a long ball intercepted at
    # midfield rather than lost at the passer's own feet).
    offensive_turnovers['End X Coord'] = np.where(
        offensive_turnovers['Event Type'] == 'dribble',
        offensive_turnovers['Player X Coord'], offensive_turnovers['Pass End X Coord'])
    offensive_turnovers['End Y Coord'] = np.where(
        offensive_turnovers['Event Type'] == 'dribble',
        offensive_turnovers['Player Y Coord'], offensive_turnovers['Pass End Y Coord'])

    valid_end = np.isfinite(offensive_turnovers['End X Coord'].astype(float)) & np.isfinite(offensive_turnovers['End Y Coord'].astype(float))
    offensive_turnovers = offensive_turnovers[valid_end]

    # xT of the opponent recovering the ball at (End X/Y Coord) — they attack toward THIS
    # team's own goal from there (x=0 in this file's own-attacking-direction convention),
    # the opposite direction from the standard xT grid orientation, hence
    # attacking_toward_100=False.
    offensive_turnovers['xT Conceded'] = lookup_xt(
        offensive_turnovers['End X Coord'].to_numpy(),
        offensive_turnovers['End Y Coord'].to_numpy(),
        attacking_toward_100=False,
    )

    # Reward side: most actions in this zone don't turn the ball over — they're
    # completed, and the team gains (a little or a lot of) xT by progressing. Without
    # this, a zone can look purely risky when it's actually net-positive because the
    # reward on the (much more frequent) successful attempts outweighs the turnover
    # cost. Passes only — successful dribbles have no end-location (Pass End X/Y Coord
    # is always NaN for dribbles), the same limitation the turnover side above already
    # works around, but there's no start-of-turn fallback for a *gain* the way there is
    # for a turnover's own position, so they're excluded here rather than approximated.
    successful_passes = offensive_turnover_actions[
        (offensive_turnover_actions['Outcome'] == True) & (offensive_turnover_actions['Event Type'] == 'pass')
    ].copy()
    valid_pass_end = np.isfinite(successful_passes['Pass End X Coord']) & np.isfinite(successful_passes['Pass End Y Coord'])
    successful_passes = successful_passes[valid_pass_end]

    # attacking_toward_100=True (not False like the turnover side) — the team still has
    # the ball, attacking their own x=100, same convention generate_pressing_turnover_
    # map's 'xT Won' uses.
    successful_passes['xT Start'] = lookup_xt(
        successful_passes['Player X Coord'].to_numpy(),
        successful_passes['Player Y Coord'].to_numpy(),
        attacking_toward_100=True,
    )
    successful_passes['xT End'] = lookup_xt(
        successful_passes['Pass End X Coord'].to_numpy(),
        successful_passes['Pass End Y Coord'].to_numpy(),
        attacking_toward_100=True,
    )
    successful_passes['xT Gained'] = successful_passes['xT End'] - successful_passes['xT Start']

    # Bin over just the defensive half (length 0-50) — pitch.bin_statistic always spans
    # the full pitch with no way to restrict its range, so this uses the same
    # custom-range approach as generate_relative_gk_pass_map for the same reason. Bins
    # keep the same physical cell size as the other charts in this file (16.67 x 14.8
    # units each), just half as many columns as the full-pitch (6,5) grid.
    #
    # Rotated so the goal sits at the top and the halfway line at the bottom: length
    # (0-50, 0=own goal) drives the screen-VERTICAL axis, width (0-100) drives the
    # screen-horizontal axis — the opposite of the other charts in this file, which are
    # all length-horizontal.
    length_bins, width_bins = 3, 5
    length_extent = (0, 50)
    width_extent = (0, 100)
    length_edges = np.linspace(*length_extent, length_bins + 1)
    width_edges = np.linspace(*width_extent, width_bins + 1)

    total_action_grid, _, _ = np.histogram2d(
        offensive_turnover_actions['Player Y Coord'], offensive_turnover_actions['Player X Coord'],
        bins=(width_bins, length_bins), range=[width_extent, length_extent])
    turnover_xt_grid, _, _ = np.histogram2d(
        offensive_turnovers['Player Y Coord'], offensive_turnovers['Player X Coord'],
        bins=(width_bins, length_bins), range=[width_extent, length_extent], weights=offensive_turnovers['xT Conceded'])
    turnover_count_grid, _, _ = np.histogram2d(
        offensive_turnovers['Player Y Coord'], offensive_turnovers['Player X Coord'],
        bins=(width_bins, length_bins), range=[width_extent, length_extent])
    gained_xt_grid, _, _ = np.histogram2d(
        successful_passes['Player Y Coord'], successful_passes['Player X Coord'],
        bins=(width_bins, length_bins), range=[width_extent, length_extent], weights=successful_passes['xT Gained'])

    # Some zones may see zero actions at all; guard against 0/0 rather than letting it warn.
    with np.errstate(invalid='ignore', divide='ignore'):
        risk_rate = np.where(total_action_grid > 0, turnover_xt_grid / total_action_grid, np.nan)
        reward_rate = np.where(total_action_grid > 0, gained_xt_grid / total_action_grid, np.nan)
        turnover_pct = np.where(total_action_grid > 0, turnover_count_grid / total_action_grid * 100, np.nan)
    # Signed from the PRESSING team's perspective (risk − reward, not reward − risk):
    # how much this zone costs the ball-playing team per action they attempt in it, i.e.
    # what a presser stands to gain there. Higher = better press target, which is the
    # direction a coach reads the chart in.
    press_value = risk_rate - reward_rate

    return {
        'press_value': press_value,
        'turnover_pct': turnover_pct,
        'total_actions': total_action_grid,
        'length_edges': length_edges,
        'width_edges': width_edges,
    }

#%%
def generate_buildup_turnover_map(leagues, team, team_matches, date, team_colors, page, bbox=[0.08, 0.0684, 0.29, 0.42]):

    team_grids = buildup_grids(team_matches)
    league_grids = buildup_grids(load_league_data(leagues, date, ['Player_Event_Data.csv'], exclude_team=team))

    # The empty guard every sibling chart in this file has, which this one was missing. BOTH sides
    # are checked: this chart is a DIFFERENCE against the league, so an empty league baseline is as
    # fatal as an empty team grid, and a competition SofaScore publishes no event data for (NCAA)
    # empties the two at once. buildup_grids returns None in that case — see there for why it
    # cannot return empty arrays instead.
    if team_grids is None or league_grids is None:
        print('No buildup data available — skipping buildup turnover map.')
        return page, None

    # Difference the per-action RATES directly — deliberately NOT converting each side to
    # its own share of total first, the way generate_defensive_action_map does. That
    # chart wants to isolate WHERE a team defends from HOW MUCH; here "how much" is the
    # entire point. A team that simply loses the ball less often than the league should
    # read as less pressable everywhere, which is exactly the "sit off, they rarely give
    # it away" call this chart exists to support. Both sides are already per-action
    # rates, so they're directly comparable without re-normalising.
    press_value = team_grids['press_value'] - league_grids['press_value']
    turnover_pct = team_grids['turnover_pct'] - league_grids['turnover_pct']
    length_edges, width_edges = team_grids['length_edges'], team_grids['width_edges']

    # Diverging around a zero that is now genuinely meaningful: the league average. The
    # structural xT offset that made an ABSOLUTE zero meaningless here (a turnover
    # forfeits a whole accumulated stock while a completion adds only a small increment,
    # so the raw quantity comes out negative in nearly every zone for every team)
    # applies equally to every team, and so cancels in the difference.
    #
    # Symmetric vmin/vmax rather than the observed min/max: scaling to each team's own
    # range made the colours within-team normalised, so a side that rarely loses the
    # ball rendered just as hot as one that constantly does. A fixed symmetric scale
    # keeps the same colour meaning the same thing from one opponent's report to the next.
    cmap = LinearSegmentedColormap.from_list('cmap', [team_colors[1], '#ffffff', team_colors[0]])

    ax = page.add_axes(bbox)
    # pitch_color='none' so the pitch's own background fill doesn't paint over the
    # heatmap when the lines get redrawn on top of it below. Draw the FULL pitch (so the
    # goal/box/halfway-line geometry is correct), then crop to just the defensive half —
    # VerticalPitch maps pitch length to the y-axis with HIGH length at the top (its own
    # half=True shows the opposite/attacking half, the wrong end for this convention), so
    # ylim is set reversed (50, 0) rather than (0, 50) to put OUR goal (length=0) at the
    # top instead.
    pitch = VerticalPitch(pitch_type='opta', pitch_color='none', line_color='#888888', linewidth=1)
    pitch.draw(ax=ax)
    ax.set_ylim(50, 0)

    # histogram2d's grid here is (width_bins, length_bins); pcolormesh wants
    # (length_bins, width_bins) since Y=length_edges, X=width_edges.
    mesh = ax.pcolormesh(width_edges, length_edges, press_value.T, cmap=cmap,
                         vmin=-BUILDUP_DIFF_SCALE, vmax=BUILDUP_DIFF_SCALE)

    # Redraw the pitch lines on top of the heatmap fill so they're visible — same
    # zorder-bump fix already used in generate_defensive_action_map (mplsoccer's line
    # artists default to zorder=0.9, below pcolormesh's zorder=1).
    existing_artists = set(ax.get_children())
    pitch.draw(ax=ax)
    ax.set_ylim(50, 0)
    for artist in ax.get_children():
        if artist not in existing_artists:
            artist.set_zorder(mesh.get_zorder() + 1)

    # A dedicated colorbar axes, sized off ax's *actual* resolved position rather
    # than colorbar(ax=ax, ...)'s default "steal space from ax" behavior — the
    # latter shrinks ax a second time, which double-triggers mplsoccer's aspect
    # enforcement and throws the colorbar/title out of alignment with the pitch.
    pitch_pos = ax.get_position()
    cbar_ax = page.add_axes([pitch_pos.x1 + 0.02, pitch_pos.y0, 0.015, pitch_pos.height])
    # extend='both': the scale is now fixed (see the *_DIFF_SCALE constants), so a cell
    # can genuinely exceed it — arrow caps flag that instead of silently flattening an
    # extreme value into the same colour as an ordinary one.
    cbar = page.colorbar(mesh, cax=cbar_ax, extend='both')
    cbar.set_label('Press Value / Action (Team − League)', labelpad=8, fontsize=8)

    # Two-line label, both values now signed diffs against league average: press value
    # on top, turnover-rate gap in percentage points below it. The signs are meaningful
    # again (unlike the previous absolute version) — positive means more pressable than
    # a typical side, negative means they hold the ball better than most there, which is
    # what supports a "sit off" call. A zone with zero recorded actions is NaN (guarded
    # in buildup_grids to avoid a 0/0 warning) rather than a real rate — labeled
    # explicitly instead of printing the literal string "nan".
    width_centers = (width_edges[:-1] + width_edges[1:]) / 2
    length_centers = (length_edges[:-1] + length_edges[1:]) / 2
    for i, wc in enumerate(width_centers):
        for j, lc in enumerate(length_centers):
            label = 'No\nActions' if np.isnan(press_value[i, j]) else f'{press_value[i, j]:+.3f}\n{turnover_pct[i, j]:+.0f}% TO'
            ax.text(wc, lc, label, color=diff_label_color(press_value[i, j], BUILDUP_DIFF_SCALE, cmap),
                     fontsize=6.5, ha='center', va='center', zorder=mesh.get_zorder() + 2)

    ax.set_title('Buildup Vulnerability vs. League Average', fontsize=10, pad=12)
    ax.set_ylabel('←   Attack    ←', labelpad=8, fontsize=8)

    # bbox[1] is the requested box's bottom edge, not where the pitch actually ends —
    # VerticalPitch's aspect lock shrinks and vertically centers the pitch within that
    # box. 0.05 is a fixed offset down from bbox[1] tuned to clear the pitch and its
    # ylabel (checked against a real render). Centered on the left half of the page
    # (x=0.25) rather than under the chart's own bbox[0]. Anchored on the same TOP edge
    # as before (old_y0 + old_height=0.06) and grown downward from there — growing from
    # y0 upward instead ate into the plot above it.
    notes_width = bbox[2] + NOTES_WIDTH_PAD
    notes_x0 = 0.25 - notes_width / 2
    notes_y0 = bbox[1] + 0.05 - 0.06
    notes_bbox = draw_notes_box(page, [notes_x0, notes_y0, notes_width, 0.12])

    return page, notes_bbox

#%%
def rotate_into_opponent_frame(frame, coord_pairs):
    '''Overlay one team's coordinates onto the other team's frame, in place.

    Each team's own Player_Event_Data.csv uses ITS OWN convention (x=0 is that team's own goal,
    x=100 the goal they attack), so the same patch of grass carries different numbers in the two
    files. The transform between them is a 180 DEGREE ROTATION — both axes flip — not a mirror of
    the length axis alone.

    THIS WAS MEASURED, and it corrects a real error. Every site below previously flipped X only,
    on the belief that "this dataset's width axis isn't flipped between the two teams' own
    frames". Pairing an event where one team lost the ball against the opposing event that won it
    — the two must be recorded at the same patch of grass — across 308 paired MLS fixtures, using
    only deliveries/losses more than 20 units off centre where a Y flip actually changes the
    answer:

        pairing                       n        X only (<2u)    rotate (<2u)
        interceptions             11,465            2.0%          12.0%
        tackles                   26,331            3.0%           8.0%
        ball-recoveries           11,465            7.2%           9.5%
        corner clearances (<3u)      209             43%            53%

    The interception row is the one that settles it: an interception is logged exactly where the
    ball was, so a genuine pair lands near zero distance, and rotation finds six times as many of
    them. Tackles and ball-recoveries agree at lower resolution — a tackle is logged at the
    tackler's feet and a recovery after the ball has run on, so both carry real physical spread
    that dilutes the signal without changing its direction.

    Under the old X-only mirror every defensive chart in this report was flipped top-to-bottom:
    "they concede down their right" rendered as their left.

    Callers must apply this AFTER any value lookup (xT, xG), never before — those grids are
    defined in the shooter's own frame. See the "value first, mirror second" note in
    generate_def_key_pass_map.'''
    for x_col, y_col in coord_pairs:
        frame[x_col] = 100 - frame[x_col]
        frame[y_col] = 100 - frame[y_col]
    return frame

#%%
def mirrored_buildout_actions(matches):
    '''Build-out actions (own defensive half) from `matches`, rotated into the frame of
    the team pressing them.

    A team's build-out half (their own X <= 50) is the same real-world zone as their opponent's
    offensive half, just in the opposite numeric frame. See rotate_into_opponent_frame for what
    the transform is and why it flips both axes rather than X alone.'''

    actions = extract_offensive_turnover_actions(matches)
    actions = actions[actions['Player X Coord'] <= 50].copy()
    rotate_into_opponent_frame(actions, [('Player X Coord', 'Player Y Coord')])
    return actions

#%%
def pressing_grids(pressing_matches, buildout_actions):
    '''Per-zone press-threat grids over the offensive half, for one pool of matches.
    Factored out of generate_pressing_turnover_map so the scouted team and the league
    baseline are computed by identical code — the chart differences the two.

    `buildout_actions` is the denominator: build-out actions already mirrored into the
    pressing team's frame (see mirrored_buildout_actions). The caller supplies it rather
    than this function deriving it, because the team and league sides source it
    differently — see generate_pressing_turnover_map.'''

    # Numerator: the scouted team's own defensive actions in their offensive half that
    # actually win the ball back cleanly. Interceptions and ball-recoveries clearly
    # count; tackles do too, since this plot is about THIS team winning the ball (unlike
    # the buildup map's turnover count, which excludes tackles for an unrelated reason —
    # an imprecise xT-loss location on a won tackle). Clearances and blocks are left out
    # — defensive contact without necessarily gaining controlled possession.
    pressing_actions = extract_def_actions(pressing_matches)
    pressing_actions = pressing_actions[pressing_actions['Player X Coord'] >= 50]
    pressing_turnovers = pressing_actions[pressing_actions['Event Sub-Type'].isin(['interception', 'tackle', 'ball-recovery'])].copy()

    # Unlike a failed pass/dribble (buildup map), a defensive action has no separate "end
    # location" — the ball is won right where the defender is, so Player X/Y Coord is
    # both the bin location and the point to value.
    #
    # xT of THIS team now attacking from (Player X/Y Coord) — after winning the ball back
    # they attack toward their own x=100 goal (this file's convention), the same
    # direction the xT grid is already oriented for, hence attacking_toward_100=True (no
    # mirroring — the opposite of the buildup map, where the OPPONENT does the attacking
    # after the scouted team loses the ball).
    pressing_turnovers['xT Won'] = lookup_xt(
        pressing_turnovers['Player X Coord'].to_numpy(),
        pressing_turnovers['Player Y Coord'].to_numpy(),
        attacking_toward_100=True,
    )

    # Bin over just the offensive half (length 50-100) — pitch.bin_statistic always spans
    # the full pitch with no way to restrict its range, so this uses the same
    # custom-range approach as generate_buildup_turnover_map for the same reason. Bins
    # keep the same physical cell size as the other charts in this file (16.67 x 14.8
    # units each), just half as many columns as the full-pitch (6,5) grid.
    #
    # Rotated so the opponent's goal sits at the top and the halfway line at the bottom:
    # length (50-100) drives the screen-VERTICAL axis, width (0-100) drives the screen-
    # horizontal axis — the opposite of the other charts in this file, which are all
    # length-horizontal.
    length_bins, width_bins = 3, 5
    length_extent = (50, 100)
    width_extent = (0, 100)
    length_edges = np.linspace(*length_extent, length_bins + 1)
    width_edges = np.linspace(*width_extent, width_bins + 1)

    total_action_grid, _, _ = np.histogram2d(
        buildout_actions['Player Y Coord'], buildout_actions['Player X Coord'],
        bins=(width_bins, length_bins), range=[width_extent, length_extent])
    turnover_xt_grid, _, _ = np.histogram2d(
        pressing_turnovers['Player Y Coord'], pressing_turnovers['Player X Coord'],
        bins=(width_bins, length_bins), range=[width_extent, length_extent], weights=pressing_turnovers['xT Won'])
    turnover_count_grid, _, _ = np.histogram2d(
        pressing_turnovers['Player Y Coord'], pressing_turnovers['Player X Coord'],
        bins=(width_bins, length_bins), range=[width_extent, length_extent])

    # Some zones may see zero actions at all; guard against 0/0 rather than letting it warn.
    with np.errstate(invalid='ignore', divide='ignore'):
        press_threat = np.where(total_action_grid > 0, turnover_xt_grid / total_action_grid, np.nan)
        win_pct = np.where(total_action_grid > 0, turnover_count_grid / total_action_grid * 100, np.nan)

    return {
        'press_threat': press_threat,
        'win_pct': win_pct,
        'total_actions': total_action_grid,
        'length_edges': length_edges,
        'width_edges': width_edges,
    }

#%%
def generate_pressing_turnover_map(leagues, team, team_matches, date, team_colors, page, bbox=[0.55, 0.0684, 0.29, 0.42]):

    # Team side denominator: the REAL opponents' own offensive activity (pass + dribble)
    # in this zone, not how much the scouted team merely defended there — normalizes
    # press value by how much the true opponent actually tried to play through it.
    # `team_matches` only holds the scouted team's own events (a match folder's
    # Player_Event_Data.csv is a single roster, never both sides), so the real opponents'
    # data for these specific fixtures is fetched separately.
    opponent_matches = load_actual_opponent_events(team_matches, leagues, date, team)
    team_grids = pressing_grids(team_matches, mirrored_buildout_actions(opponent_matches))

    # League side denominator does NOT need that per-fixture opponent matching. Pooled
    # over every team in the league, "build-out actions faced" and "build-out actions
    # performed" are the same multiset — each match contributes both teams' build-out
    # actions either way — so the league pool's own mirrored build-out actions are
    # already the correct denominator. This relies on the league being closed (every
    # opponent also having a scraped folder of their own), which was verified on this
    # dataset. It's also dramatically cheaper than calling load_actual_opponent_events
    # once per team.
    league_matches = load_league_data(leagues, date, ['Player_Event_Data.csv'], exclude_team=team)
    league_grids = pressing_grids(league_matches, mirrored_buildout_actions(league_matches))

    # Difference the per-action rates directly (see generate_buildup_turnover_map for why
    # these are NOT re-normalised to shares first).
    press_threat = team_grids['press_threat'] - league_grids['press_threat']
    press_win_pct = team_grids['win_pct'] - league_grids['win_pct']
    length_edges, width_edges = team_grids['length_edges'], team_grids['width_edges']

    # No sign flip here, unlike generate_buildup_turnover_map — that chart's raw quantity
    # was the ball-playing team's own value and had to be negated to read as the
    # presser's gain. This chart scouts the OPPOSITION's press (generateSecondPage passes
    # `opposition` as `team`), so a high value already means they win the ball back
    # dangerously there, i.e. a zone to avoid building through. Positive now means they
    # press that zone more dangerously than a typical side does; negative means it's a
    # relatively safe area to play through against them.
    cmap = LinearSegmentedColormap.from_list('cmap', [team_colors[1], '#ffffff', team_colors[0]])

    ax = page.add_axes(bbox)
    # pitch_color='none' so the pitch's own background fill doesn't paint over the
    # heatmap when the lines get redrawn on top of it below. Draw the FULL pitch (so the
    # goal/box/halfway-line geometry is correct), then crop to just the offensive half —
    # VerticalPitch's default already puts HIGH length (100, the opponent's goal) at the
    # top, which is exactly what's wanted here, so unlike generate_buildup_turnover_map
    # (which needs a reversed ylim to put length=0 at the top instead) this crop uses the
    # default direction as-is.
    pitch = VerticalPitch(pitch_type='opta', pitch_color='none', line_color='#888888', linewidth=1)
    pitch.draw(ax=ax)
    ax.set_ylim(50, 100)

    # histogram2d's grid here is (width_bins, length_bins); pcolormesh wants
    # (length_bins, width_bins) since Y=length_edges, X=width_edges.
    mesh = ax.pcolormesh(width_edges, length_edges, press_threat.T, cmap=cmap,
                         vmin=-PRESSING_DIFF_SCALE, vmax=PRESSING_DIFF_SCALE)

    # Redraw the pitch lines on top of the heatmap fill so they're visible — same
    # zorder-bump fix already used in generate_defensive_action_map (mplsoccer's line
    # artists default to zorder=0.9, below pcolormesh's zorder=1).
    existing_artists = set(ax.get_children())
    pitch.draw(ax=ax)
    ax.set_ylim(50, 100)
    for artist in ax.get_children():
        if artist not in existing_artists:
            artist.set_zorder(mesh.get_zorder() + 1)

    # A dedicated colorbar axes, sized off ax's *actual* resolved position rather
    # than colorbar(ax=ax, ...)'s default "steal space from ax" behavior — the
    # latter shrinks ax a second time, which double-triggers mplsoccer's aspect
    # enforcement and throws the colorbar/title out of alignment with the pitch.
    pitch_pos = ax.get_position()
    cbar_ax = page.add_axes([pitch_pos.x1 + 0.02, pitch_pos.y0, 0.015, pitch_pos.height])
    # extend='both': the scale is now fixed (see the *_DIFF_SCALE constants), so a cell
    # can genuinely exceed it — arrow caps flag that instead of silently flattening an
    # extreme value into the same colour as an ordinary one.
    cbar = page.colorbar(mesh, cax=cbar_ax, extend='both')
    cbar.set_label('Press Threat / Action (Team − League)', labelpad=8, fontsize=8)

    # Two-line label mirroring generate_buildup_turnover_map's: press threat on top, and
    # below it the share of build-out actions in that zone the opposition actually wins
    # — which separates a zone they win rarely but devastatingly from one they win often
    # and cheaply. Unsigned '{:.3f}' (values are small non-negative decimals and the
    # scale is relative). A zone with zero recorded actions is NaN (guarded above to
    # avoid a 0/0 warning) rather than a real rate — labeled explicitly instead of
    # printing the literal string "nan".
    width_centers = (width_edges[:-1] + width_edges[1:]) / 2
    length_centers = (length_edges[:-1] + length_edges[1:]) / 2
    for i, wc in enumerate(width_centers):
        for j, lc in enumerate(length_centers):
            label = 'No\nActions' if np.isnan(press_threat[i, j]) else f'{press_threat[i, j]:+.3f}\n{press_win_pct[i, j]:+.0f}% Won'
            ax.text(wc, lc, label, color=diff_label_color(press_threat[i, j], PRESSING_DIFF_SCALE, cmap),
                        fontsize=6.5, ha='center', va='center', zorder=mesh.get_zorder() + 2)

    ax.set_title('Press Threat vs. League Average', fontsize=10, pad=12)
    ax.set_ylabel('→   Attack    →', labelpad=8, fontsize=8)

    # bbox[1] is the requested box's bottom edge, not where the pitch actually ends —
    # VerticalPitch's aspect lock shrinks and vertically centers the pitch within that
    # box. 0.05 is a fixed offset down from bbox[1] tuned to clear the pitch and its
    # ylabel (checked against a real render). Centered on the right half of the page
    # (x=0.75), mirroring the left-side charts. Anchored on the same TOP edge as before
    # (old_y0 + old_height=0.06) and grown downward from there — growing from y0 upward
    # instead ate into the plot above it.
    notes_width = bbox[2] + NOTES_WIDTH_PAD
    notes_x0 = 0.75 - notes_width / 2
    notes_y0 = bbox[1] + 0.05 - 0.06
    notes_bbox = draw_notes_box(page, [notes_x0, notes_y0, notes_width, 0.12])

    return page, notes_bbox

#%%
def from_corner_arc(events):
    '''Boolean mask: events struck from the standardised corner-arc placeholder.

    Corner kicks come through with the same Event Type as open-play crosses/passes — no
    situation or set-piece flag exists anywhere in Player_Event_Data.csv — but they are struck
    from a standardised placeholder coordinate right at the arc rather than a real tracked
    position. Confirmed empirically: of all crosses near a corner arc, 79% sit at just two
    exactly-repeated coordinates (99.5/0.5 and 99.5/99.5), versus organically varied positions
    for genuine open-play byline deliveries.

    One definition, shared by the chart that throws corners AWAY (generate_key_pass_map, via
    extract_key_passes) and the charts that are ABOUT them (extract_corners), so the two can
    never disagree about what a corner is.'''
    return (events['Player X Coord'] > 99) & ((events['Player Y Coord'] < 1) | (events['Player Y Coord'] > 99))

#%%
# Every column extract_corners produces, so the empty path returns a frame its callers can
# filter and group against without KeyError-ing. The last three are derived here.
CORNER_COLUMNS = ['Match', 'Player', 'Event Type', 'Key Pass', 'Outcome',
                  'Player X Coord', 'Player Y Coord', 'Pass End X Coord', 'Pass End Y Coord',
                  'Arc', 'Completed']

def extract_corners(matches):
    '''Every corner delivery across `matches`, in the taker's own frame.

    Unlike extract_key_passes this keeps MATCH IDENTITY in a 'Match' column. Both things the
    corner charts do with these rows — linking a delivery to the shot it created, and linking a
    conceded delivery to the defender who cleared it — are per-match joins, and a plain concat
    would silently match a delivery against a shot from a different fixture.

    Derived columns:
      Arc       - which corner arc it was struck from, as a bool (Player Y Coord < 1).
      Completed - whether it found a teammate. Outcome is True/NaN in this file, never False,
                  so a NaN IS an incomplete delivery rather than missing data. Compared against
                  True rather than filled: if a False ever does appear it lands on the right
                  side either way, and the object column is never downcast.'''
    matches_corners = []
    for match, match_files in matches.items():
        event_data = match_files.get('Player_Event_Data.csv')
        if event_data is None:
            continue
        match_corners = event_data[from_corner_arc(event_data)].copy()
        match_corners['Match'] = match
        matches_corners.append(match_corners)

    if not matches_corners:
        return pd.DataFrame(columns=CORNER_COLUMNS)

    corners = pd.concat(matches_corners, ignore_index=True)
    corners['Arc'] = corners['Player Y Coord'] < 1
    corners['Completed'] = corners['Outcome'].eq(True)

    return corners

# How close a corner shot must sit to a delivery's endpoint to count as the shot that delivery
# created. Measured: across 25 key-pass corners the nearest corner-situation shot sat at a median
# of 0.0 opta units and a max of 2.7, because SofaScore records the delivery's end AT the shot's
# own location. 3 is comfortably above that ceiling and far below the spacing between distinct
# shots, so the guard rejects non-matches without ever splitting a real pair.
CORNER_SHOT_MATCH_RADIUS = 3

# The same idea for the defensive side, but looser, and the difference is physical rather than
# arbitrary: a clearance is logged where the DEFENDER was standing, not where the ball was, so a
# genuine pair carries a couple of units of real spread (measured median 1-3, against 0.0 for the
# shot link). 5 catches 93% of incomplete conceded corners. Attribution here is good, not exact.
CORNER_DEFENSIVE_MATCH_RADIUS = 5

def ids_by_folder(matches):
    '''{folder name: SofaScore match id} for `matches`. The inverse of _matches_by_id, and needed
    for the same reason: folder names are NOT a join key across two teams' copies of a fixture.'''
    ids = {}
    for folder, match_files in matches.items():
        attributes = match_files.get('Match_Attributes.csv')
        if attributes is not None and 'id' in attributes.columns:
            ids[folder] = attributes['id'].iloc[0]
    return ids

def corner_xg_created(corners, corner_matches, matches, team, conceded=False):
    '''xG of the shot each delivery in `corners` created, aligned to `corners`, 0.0 where none.

    `corner_matches` is the dict `corners` were extracted from and `matches` the scouted team's
    own. For corners TAKEN they are the same dict; for corners CONCEDED they are not, and the two
    disagree about folder names — a fixture is filed under the other side's name in each team's
    directory. Joining on the folder string silently matched nothing and scored every conceded
    delivery 0.0, so the join goes through the match ID, as every other cross-team join here does.

    FIRST CONTACT ONLY, by construction, and the charts must say so. Only deliveries flagged
    Key Pass are considered, which is the direct assist — a corner that is half-cleared and
    hammered in from the edge produces a corner-situation shot with no key pass at the arc. On
    Nashville that is 49 corner shots against 25 key-pass corners, so roughly half the corner
    threat is second phase and is deliberately outside this number. It is why the heatmap this
    feeds and the shooter list beside it do not total the same xG.

    Joined per match on `corners['Match']` — see extract_corners for why that column exists.

    A key-pass corner in a competition SofaScore does not model xG for scores 0.0 rather than
    linking, because extract_shots has already dropped the shot it created. That is correct and
    not a miss: on Nashville exactly one of 25 falls in this bucket (a Leagues Cup tie whose
    Shot_Map carries no finite xG at all), and all 24 inside xG-carrying matches link.'''
    created = pd.Series(0.0, index=corners.index)
    if corners.empty:
        return created

    shots = extract_shots(matches, team, conceded=conceded, quiet=True)
    if shots.empty:
        return created
    shots = shots[shots['situation'] == 'corner']

    shot_ids = ids_by_folder(matches)
    corner_ids = ids_by_folder(corner_matches)
    shots_by_id = {shot_ids[folder]: group for folder, group in shots.groupby('Match')
                   if folder in shot_ids}

    for i, delivery in corners[corners['Key Pass'] == True].iterrows():
        match_shots = shots_by_id.get(corner_ids.get(delivery['Match']))
        if match_shots is None:
            continue
        distance = np.hypot(match_shots['Shot X Coord'] - delivery['Pass End X Coord'],
                            match_shots['Shot Y Coord'] - delivery['Pass End Y Coord'])
        if distance.min() <= CORNER_SHOT_MATCH_RADIUS:
            created.loc[i] = match_shots.loc[distance.idxmin(), 'xg']

    return created

def _matches_by_id(matches):
    '''{SofaScore match id: match_files} for `matches`. Folder NAMES can't be joined across two
    teams' copies of the same fixture — they're each named after the other side, and the same
    match can carry different dates depending on scrape-time timezone (see loadMatchData) — so
    every cross-team join in this file goes through the match ID, as load_actual_opponent_events
    already does.'''
    by_id = {}
    for match_files in matches.values():
        attributes = match_files.get('Match_Attributes.csv')
        if attributes is not None and 'id' in attributes.columns:
            by_id[attributes['id'].iloc[0]] = match_files
    return by_id

def corner_defensive_actions(conceded_corners, opponent_matches, matches):
    '''Who dealt with the corners this team faced: {player: actions}, counted over the corners
    their defence actually won.

    Only INCOMPLETE conceded corners are considered, which is the honest scope rather than a
    convenience: a corner that found an attacker was not defended, so there is no defender to
    credit for it. The list therefore reads as "who wins the ball", and failing to win it
    correctly earns nobody a mention.

    ATTRIBUTION IS GOOD, NOT EXACT, and the chart label must not overclaim. A clearance is
    logged where the DEFENDER was standing rather than where the ball was, so a genuine pair
    carries a couple of units of real spread — measured median 1-3 units, against 0.0 for the
    delivery-to-shot link, which is why this uses its own looser radius. It reaches 93% of
    incomplete conceded corners, and the subtype mix is what it should be: on Nashville,
    29 clearances, 24 ball-recoveries, 5 blocks, 1 interception.

    `conceded_corners` are in the DELIVERING team's frame; the scouted team's own defensive
    actions are in theirs, so they are rotated across before measuring (see
    rotate_into_opponent_frame — this is the join the rotation was established for).'''
    counts = {}
    if conceded_corners.empty:
        return counts

    own_by_id = _matches_by_id(matches)
    # The corners' 'Match' is an OPPONENT folder name, so map it to the shared match ID first.
    opponent_id_by_folder = ids_by_folder(opponent_matches)

    rotated_actions = {}
    for _, delivery in conceded_corners[~conceded_corners['Completed']].iterrows():
        match_id = opponent_id_by_folder.get(delivery['Match'])
        if match_id is None:
            continue
        if match_id not in rotated_actions:
            own_files = own_by_id.get(match_id)
            events = None if own_files is None else own_files.get('Player_Event_Data.csv')
            if events is None:
                rotated_actions[match_id] = None
            else:
                actions = events[events['Event Type'] == 'Defensive Action'].copy()
                rotated_actions[match_id] = rotate_into_opponent_frame(
                    actions, [('Player X Coord', 'Player Y Coord')]) if not actions.empty else None
        actions = rotated_actions[match_id]
        if actions is None:
            continue

        distance = np.hypot(actions['Player X Coord'] - delivery['Pass End X Coord'],
                            actions['Player Y Coord'] - delivery['Pass End Y Coord'])
        if distance.min() <= CORNER_DEFENSIVE_MATCH_RADIUS:
            player = actions.loc[distance.idxmin(), 'Player']
            counts[player] = counts.get(player, 0) + 1

    return counts

#%%
def extract_key_passes(matches):

    matches_key_passes = []
    for match, match_files in matches.items():
        if 'Player_Event_Data.csv' not in match_files:
            continue
        event_data = match_files['Player_Event_Data.csv']
        match_key_passes = event_data[event_data['Key Pass'] == True]
        matches_key_passes.append(match_key_passes)

    if not matches_key_passes:
        return pd.DataFrame(columns=['Player', 'Key Pass', 'Player X Coord', 'Player Y Coord', 'Pass End X Coord', 'Pass End Y Coord'])

    key_passes = pd.concat(matches_key_passes)

    # Excluded here (not just in the chart) so every downstream user — the heatmap, the scatter
    # dots, and the per-90 counts — is limited to open play. See from_corner_arc.
    key_passes = key_passes[~from_corner_arc(key_passes)]

    return key_passes

#%%
# bbox[0]=0.03 sits the pitch's left edge just inside the page's inner border, which
# generatePageTemplate puts at (6+5)pt / (72*8.5in) = 0.018 in figure fractions. The
# pitch is width-constrained inside this bbox (its aspect lock shrinks the axes
# vertically, not horizontally), so the drawn pitch starts at bbox[0] itself.
def generate_key_pass_map(leagues, team, team_matches, date, team_colors, page,
                          bbox=[0.03, 0.50, 0.26, 0.42],
                          expected_lineup=None, expected_subs=None):

    key_passes = extract_key_passes(team_matches)

    if key_passes.empty:
        print('No key pass data available — skipping key pass map.')
        return page, None, None

    valid_end = np.isfinite(key_passes['Pass End X Coord']) & np.isfinite(key_passes['Pass End Y Coord'])
    key_passes = key_passes[valid_end].copy()

    ax = page.add_axes(bbox)
    # pitch_color='none' so the pitch's own background matches the page instead of
    # painting a separate fill behind it — same as every other chart in this file.
    pitch = Pitch(pitch_type='opta', pitch_color='none', line_color='#888888', linewidth=1)
    pitch.draw(ax=ax)

    # xT of where THIS team's key pass was delivered to — after playing it, they're
    # still attacking toward their own x=100 goal (this file's convention), the same
    # direction the xT grid is already oriented for, hence attacking_toward_100=True (no
    # mirroring — same reasoning as generate_pressing_turnover_map's 'xT Won').
    key_passes['xT Delivered'] = lookup_xt(
        key_passes['Pass End X Coord'].to_numpy(),
        key_passes['Pass End Y Coord'].to_numpy(),
        attacking_toward_100=True,
    )

    # Background: binned by ORIGIN (Player X/Y Coord — where the passer had to be to
    # create it), valued by DELIVERY (Pass End Coord — how dangerous the resulting
    # position was). Answers "how much delivered danger has originated from this zone" —
    # an absolute magnitude, not diffed against league average (same reasoning
    # generate_buildup_turnover_map already uses for its own xT rate: this is a
    # self-normalized metric with meaningful units, and a prevention decision needs
    # absolute risk, not relative-to-league risk). Full pitch, same (6,5) grid
    # generate_defensive_action_map uses — key passes aren't restricted to one half the
    # way the turnover charts are.
    xt_bins = pitch.bin_statistic(key_passes['Player X Coord'], key_passes['Player Y Coord'],
                                   values=key_passes['xT Delivered'], statistic='sum', bins=(6, 5))

    cmap = LinearSegmentedColormap.from_list('cmap', ['#ffffff', team_colors[0]])

    # ONE scale across the pair, computed here because this chart can reach both directions and
    # its conceded sibling cannot. Self-normalising made a solid cell here mean a different ΣxT
    # from a solid cell on the chart beside it, so the two could not be read against each other.
    # Binned UNROTATED on purpose: rotating maps cell (i,j) to (N-i, M-j), leaving the set of cell
    # values — and so the maximum — unchanged. load_actual_opponent_events is cached in-process,
    # so the conceded chart calling it again costs nothing.
    conceded_passes = extract_key_passes(load_actual_opponent_events(team_matches, leagues, date, team))
    if not conceded_passes.empty:
        valid = (np.isfinite(conceded_passes['Pass End X Coord'])
                 & np.isfinite(conceded_passes['Pass End Y Coord']))
        conceded_passes = conceded_passes[valid]
    max_val = np.nanmax(xt_bins['statistic'])
    if not conceded_passes.empty:
        conceded_bins = pitch.bin_statistic(
            conceded_passes['Player X Coord'], conceded_passes['Player Y Coord'],
            values=lookup_xt(conceded_passes['Pass End X Coord'].to_numpy(),
                             conceded_passes['Pass End Y Coord'].to_numpy(),
                             attacking_toward_100=True),
            statistic='sum', bins=(6, 5))
        max_val = np.nanmax([max_val, np.nanmax(conceded_bins['statistic'])])
    if not max_val or not np.isfinite(max_val):
        max_val = 1
    shared_vmax = max_val

    # No per-cell numeric labels on this one — the scatter dots on top already carry
    # per-pass detail, and stacking text under them would clutter the chart rather than
    # clarify it.
    mesh = pitch.heatmap(xt_bins, ax=ax, cmap=cmap, vmin=0, vmax=max_val, zorder=1)

    # Redraw the pitch lines on top of the heatmap fill so they're visible — same
    # zorder-bump fix used by every other heatmap chart in this file.
    existing_artists = set(ax.get_children())
    pitch.draw(ax=ax)
    for artist in ax.get_children():
        if artist not in existing_artists:
            artist.set_zorder(mesh.get_zorder() + 1)

    # One player at a time (rather than a single scatter with a c= color array) so each gets its
    # own colour and its own swatch in the key. distinct_player_colors() covers a realistic number
    # of distinct key-pass takers across a date range; colors cycle rather than crash if a squad
    # ever produces more. Sorted by threat per 90 descending (not alphabetically) so the most
    # creative players surface at the top.
    player_minutes = compute_player_minutes(team_matches)

    def player_xt(player):
        '''Total xT this player has DELIVERED — the same per-pass value the heatmap sums, so the
        key and the chart underneath it are denominated in one currency.'''
        return key_passes.loc[key_passes['Player'] == player, 'xT Delivered'].sum()

    def xt_per_90(player):
        '''The ranking metric: threat delivered per 90.

        Threat-weighted rather than a raw key-pass rate, because a squared ball to the six-yard
        box and a hopeful cross to the back post both count once as a key pass. Expected ASSISTS
        would be the natural choice and is not available: linking a key pass to the shot it
        created works exactly for corners (the ball is struck at the delivery point) but only 57%
        of the time in open play, where a touch usually intervenes — measured median 1.50 units,
        p75 5.62 — so an xA column would score 43% of real chances created as zero. xT is read off
        a grid, needs no such link, and cannot silently drop anybody.'''
        minutes = player_minutes.get(player, 0)
        return player_xt(player) / minutes * 90 if minutes else 0

    players = sorted(key_passes['Player'].dropna().unique(), key=xt_per_90, reverse=True)
    named_players, other_players = players[:KEY_PASS_LEGEND_MAX], players[KEY_PASS_LEGEND_MAX:]

    qualified = player_qualifier(team_matches, expected_lineup, expected_subs, player_minutes)

    unqualified_labels = []
    player_handles, player_values = [], []

    distinct_colors = distinct_player_colors()

    for i, player in enumerate(named_players):
        player_passes = key_passes[key_passes['Player'] == player]
        # Minutes are NO LONGER a column. The greying above already says "thin sample", and the
        # space it freed buys the threat rate — which is what the key is for. The same three-part
        # shape the shot and corner keys use: rate per 90, raw volume, then the quality of what
        # the player produces.
        color = distinct_colors[i % len(distinct_colors)]
        count = len(player_passes)
        player_values.append((abbreviate_player_name(player), f'{xt_per_90(player):.3f}', count,
                              f'{player_xt(player) / count:.3f}' if count else '—'))
        player_handles.append(Patch(facecolor=color, edgecolor='black', linewidth=0.5))
        # Circles at the pass ORIGIN (Player X/Y Coord) — not the end location — per the
        # requested design: this chart is about who creates chances and from where, not
        # where they end up (already shown implicitly by "key pass" leading to a shot).
        ax.scatter(player_passes['Player X Coord'], player_passes['Player Y Coord'],
                   color=color, s=45, edgecolors='black',
                   linewidths=0.5, alpha=0.65, zorder=3)
        unqualified_labels.append(not qualified(player))

    # Everyone past the cap still gets their passes plotted — dropping them would remove
    # real events from the chart — but shares one grey swatch and a single legend row, so
    # a deep squad can't grow the key without bound. They're the lowest per-90 creators by
    # construction, so the detail lost is the least interesting.
    if other_players:
        other_passes = key_passes[key_passes['Player'].isin(other_players)]
        ax.scatter(other_passes['Player X Coord'], other_passes['Player Y Coord'],
                   color='#999999', s=45, edgecolors='black', linewidths=0.5,
                   alpha=0.65, zorder=3)
        player_handles.append(Patch(facecolor='#999999', edgecolor='black', linewidth=0.5))
        player_values.append((f'Other ({len(other_players)})', '', len(other_passes), ''))

    # A dedicated colorbar axes close against the pitch (same pattern as every other
    # chart in this file), with the player legend positioned further out so the two
    # side-by-side "keys" don't overlap.
    pitch_pos = ax.get_position()
    cbar_ax = page.add_axes([pitch_pos.x1 + 0.02, pitch_pos.y0, 0.015, pitch_pos.height])
    cbar = page.colorbar(mesh, cax=cbar_ax)
    cbar.set_label('ΣxT Delivered / Zone', labelpad=6, fontsize=8)

    # Set before the legend below, which measures the title to align itself against it.
    title_fontsize, title_pad = 10, 12
    ax.set_title('Key Passes Completed', fontsize=title_fontsize, pad=title_pad)
    ax.set_xlabel('→    Attack    →', labelpad=8, fontsize=8)

    # Two things pin the legend here:
    #
    # 'upper left' rather than 'center left' — a vertically centered legend drifts with
    # the number of players in the key-pass list, so its top edge only lines up with the
    # title by coincidence and moves from one opponent to the next. Anchoring the top
    # fixes it regardless of how many entries there are.
    #
    # ...and the anchor sits ABOVE y=1.0, because y=1.0 is the axes' top edge, which is
    # where the pitch ends, not where the title sits — the title is `pad` points further
    # up. Converting that pad (plus ~0.8 * fontsize to reach the top of the capitals)
    # out of points and into axes fractions puts the two tops level, and keeps them level
    # if the figure size or font sizes change. get_position() is the aspect-resolved box
    # by this point, same assumption the colorbar placement above already relies on.
    axes_height_points = ax.get_position().height * page.get_figheight() * 72
    legend_top = 1.0 + (title_pad + 0.8 * title_fontsize) / axes_height_points

    # Drawn as a TABLE, like every other player key on the report — the numbers are what a reader
    # scans down, and in a "Name (1.23, 4)" list they start at a different x on every row.
    #
    # Greying goes to the row TEXT rather than the swatch, deliberately: the swatch colour is what
    # ties a row to its dots on the pitch, so recolouring it would break that link. The trailing
    # "Other" row (when present) has no entry in unqualified_labels, so table_legend's zip stops
    # before it.
    #
    # Three decimals on both xT columns, where the shot key uses two on its xG ones: delivered xT
    # per 90 spans roughly 0.06-0.22 across a squad and xT per pass 0.07-0.13, so two decimals
    # collapse most of a squad onto the same two or three values.
    player_header, player_rows = format_key_table(
        [('Player', 17, '<'), ('xT/90', 6, '>'), ('KP', 3, '>'), ('xT/KP', 6, '>')], player_values)
    # Same anchor constant as the shot map's, which only lands the two key columns at the same
    # figure x because page three's two left-hand charts are now the same width. They were not:
    # this pair was 0.28 wide against the shot pair's 0.26, so an identical axes fraction put this
    # column 0.014 further right and ran it over the conceded chart at x=0.600.
    table_legend(ax, player_header, player_rows, 'Playmaking Threats', SHOT_LEGEND_ANCHOR_X, legend_top,
                 handles=player_handles, grey=unqualified_labels, handlelength=1.0)

    # bbox[1] is the requested box's bottom edge, not where the pitch actually ends —
    # mplsoccer's aspect lock shrinks and vertically centers the pitch within that box.
    # 0.05 is a fixed offset down from bbox[1] tuned to clear the pitch and its xlabel
    # (checked against a real render, same value used by the sibling turnover charts).
    # Anchored on the same TOP edge as before (old_y0 + old_height=0.06) and grown
    # downward from there — growing from y0 upward instead ate into the plot above it.
    #
    # Aligned to the chart itself rather than centred on the page's left half (x=0.25)
    # like the page-two charts: once this chart moved out to bbox[0]=0.03 the page-half
    # centre left the box visibly offset to the right of the pitch above it. Matching the
    # chart's own x-extent also avoids the NOTES_WIDTH_PAD overhang, which at this x
    # would have pushed the box's left edge to 0.01 — outside the page's inner border
    # at 0.018.
    notes_x0 = bbox[0]
    notes_y0 = bbox[1] + 0.05 - 0.06
    notes_bbox = draw_notes_box(page, [notes_x0, notes_y0, NOTES_WIDTH, 0.12])

    return page, notes_bbox, shared_vmax

#%%
# bbox[0]=0.60 clears the offensive sibling's player legend, which extends to ~0.588 —
# at 0.53 the legend ran straight over this chart's pitch and title. The chart plus its
# colourbar and rotated label occupy ~0.345, so 0.60 still finishes inside the page's
# inner border at 0.982.
def generate_def_key_pass_map(leagues, team, team_matches, date, team_colors, page,
                              bbox=[0.60, 0.50, 0.26, 0.42], shared_vmax=None):
    '''Where the scouted team CONCEDES key passes — the defensive mirror of
    generate_key_pass_map.'''

    # A team's own Player_Event_Data.csv holds only their own events, so the key passes
    # played *against* them live in their opponents' files — the same cross-reference
    # generate_pressing_turnover_map relies on for its denominator.
    opponent_matches = load_actual_opponent_events(team_matches, leagues, date, team)
    key_passes = extract_key_passes(opponent_matches)

    if key_passes.empty:
        print('No conceded key pass data available — skipping defensive key pass map.')
        return page, None

    valid_end = np.isfinite(key_passes['Pass End X Coord']) & np.isfinite(key_passes['Pass End Y Coord'])
    key_passes = key_passes[valid_end].copy()

    # Value FIRST, mirror second — the order matters. These coordinates are still in each
    # opponent's own frame (they attack toward their own x=100), so the delivery's threat
    # is the same attacking_toward_100=True lookup the offensive sibling makes. Only after
    # that are the coordinates mirrored for display. Mirroring first and compensating with
    # attacking_toward_100=False yields an identical number, but this ordering is much
    # harder to get subtly wrong later.
    key_passes['xT Delivered'] = lookup_xt(
        key_passes['Pass End X Coord'].to_numpy(),
        key_passes['Pass End Y Coord'].to_numpy(),
        attacking_toward_100=True,
    )

    # Rotate into the SCOUTED team's frame so this chart and its offensive sibling share
    # one coordinate system: the scouted team's own goal at x=0, the opposition attacking
    # leftward toward it. Chances they create then cluster right and chances they concede
    # cluster left, which is what makes the pair readable side by side.
    #
    # BOTH axes flip — see rotate_into_opponent_frame for the measurement that establishes it.
    # The xT lookup above runs first, in the passer's own frame, which is the order that matters:
    # value first, rotate second.
    #
    # extract_key_passes' corner-kick filter keys off X > 99 in the passer's OWN frame, so
    # it has already run correctly above, before this.
    rotate_into_opponent_frame(key_passes, [('Player X Coord', 'Player Y Coord'),
                                            ('Pass End X Coord', 'Pass End Y Coord')])

    ax = page.add_axes(bbox)
    # pitch_color='none' so the pitch's own background matches the page instead of
    # painting a separate fill behind it — same as every other chart in this file.
    pitch = Pitch(pitch_type='opta', pitch_color='none', line_color='#888888', linewidth=1)
    pitch.draw(ax=ax)

    # Binned by ORIGIN (where the opponent passer was), valued by DELIVERY (how dangerous
    # the position they found was) — the same construction as the offensive sibling, and
    # absolute rather than league-diffed for the same reason.
    xt_bins = pitch.bin_statistic(key_passes['Player X Coord'], key_passes['Player Y Coord'],
                                   values=key_passes['xT Delivered'], statistic='sum', bins=(6, 5))

    cmap = LinearSegmentedColormap.from_list('cmap', ['#ffffff', team_colors[0]])
    # From the offensive sibling, which is the only chart of the pair that sees both directions.
    max_val = shared_vmax if shared_vmax else np.nanmax(xt_bins['statistic'])
    if not max_val or not np.isfinite(max_val):
        max_val = 1

    mesh = pitch.heatmap(xt_bins, ax=ax, cmap=cmap, vmin=0, vmax=max_val, zorder=1)

    # Redraw the pitch lines on top of the heatmap fill so they're visible — same
    # zorder-bump fix used by every other heatmap chart in this file.
    existing_artists = set(ax.get_children())
    pitch.draw(ax=ax)
    for artist in ax.get_children():
        if artist not in existing_artists:
            artist.set_zorder(mesh.get_zorder() + 1)

    # One flat colour and no legend, unlike the offensive sibling: these passes come from
    # whichever opponents the team happened to face (12 players across 3 different clubs
    # in the Pumas sample), so a per-player key would be listing strangers from unrelated
    # teams rather than a squad. team_colors[1] over a team_colors[0] heatmap is the same
    # contrast assumption the diverging maps elsewhere in this file already make.
    # alpha=0.45 rather than the sibling's 0.8: those dots are spread across a tab20
    # palette, but these are all one colour, so at high opacity they read as a solid mass
    # that swamps the heatmap underneath — which is the layer carrying the actual xT
    # magnitude. The lower opacity also makes overlapping dots legible as clusters.
    ax.scatter(key_passes['Player X Coord'], key_passes['Player Y Coord'],
               color=team_colors[1], s=45, edgecolors='black', linewidths=0.4,
               alpha=0.45, zorder=3)

    pitch_pos = ax.get_position()
    cbar_ax = page.add_axes([pitch_pos.x1 + 0.02, pitch_pos.y0, 0.015, pitch_pos.height])
    cbar = page.colorbar(mesh, cax=cbar_ax)
    cbar.set_label('ΣxT Conceded / Zone', labelpad=6, fontsize=8)

    ax.set_title('Key Passes Conceded', fontsize=10, pad=12)
    # Arrow points LEFT because the opposition attacks toward x=0 in this mirrored frame —
    # the visible cue that this is the defensive mirror of the chart beside it.
    ax.set_xlabel('←    Opposition Attack    ←', labelpad=8, fontsize=8)

    # Centered on the right half of the page (x=0.75), mirroring the left-side sibling —
    # the same convention the page-two charts use for their notes boxes.
    notes_width = bbox[2] + NOTES_WIDTH_PAD
    notes_x0 = 0.75 - notes_width / 2
    notes_y0 = bbox[1] + 0.05 - 0.06
    notes_bbox = draw_notes_box(page, [notes_x0, notes_y0, notes_width, 0.12])

    return page, notes_bbox

#%%
# Every column generate_shot_map reads, so the empty-data path returns a frame it can filter
# and group against without KeyError-ing. The last three are derived by extract_shots itself,
# not present in Shot_Map.csv.
SHOT_COLUMNS = ['Match', 'player.name', 'situation', 'shotType', 'xg', 'xgot',
                'Shot X Coord', 'Shot Y Coord', 'Situation Group']

def extract_shots(matches, team, conceded=False, quiet=False):
    '''Every shot `team` took across `matches`, with coordinates converted into the same opta
    frame the rest of this file uses (own goal at x=0, attacking toward x=100).

    conceded=True returns the shots taken AGAINST them instead. No extra loading is needed for
    that, and deliberately so: Shot_Map.csv holds BOTH sides of a fixture and separates them by
    an isHome flag alone, so the opposition's shots are already sitting in the files this
    function is reading — inverting one comparison is the whole change. (This is the same
    property league_situation_xg_rates relies on, and why its docstring warns that
    load_league_data's `exclude_team` cannot filter this file.) The defensive key pass map has to
    cross-reference opponents' own event files via load_actual_opponent_events because
    Player_Event_Data.csv holds one team only; reaching for that here would be redundant work.

    Every filter below is side-agnostic and so needs no adjustment: own goals and shots with no
    xG are wrong to keep whoever took them, and the coordinate flip and situation mapping are
    properties of the file format, not of which team is being read. Note that the coordinates
    come back in the SHOOTER's own frame either way — a conceded shot is described as the
    opponent who took it saw it, so a caller wanting it in `team`'s frame must mirror X itself
    (see generate_def_shot_map).

    THE COORDINATE CONVERSION IS THE POINT OF THIS FUNCTION. Shot_Map.csv does NOT share
    Player_Event_Data.csv's frame, despite both being 0-100:

      * playerCoordinates.x is DISTANCE FROM THE GOAL BEING ATTACKED, not position along the
        pitch — its season-wide max is 48.2, and every penalty sits at exactly 11.5. Verified
        by pairing key-pass end locations against 'assisted' shots across all 30 MLS teams:
        key-pass ends average x=81.0, and 100 - playerCoordinates.x averages 85.3 (shots land
        slightly closer to goal than the pass that created them, as they should). The raw
        column averages 14.7, i.e. it would put every shot in the team's own defensive third.

      * playerCoordinates.y is MIRRORED relative to Player Y Coord. Per-team mean y of
        key-pass ends vs. assisted shots correlates -0.72 across those same 30 teams (40+
        events each) — real left/right attacking asymmetry pointing opposite ways in the two
        files. The draw.* columns say the same thing independently: draw.start.x equals
        playerCoordinates.y untouched, while draw.goal.x equals 100 - goalMouthCoordinates.y.

    Both axes are therefore flipped, and the results land on the same grid every other chart
    in this file bins against.

    Own goals and shots with no xG are dropped — see the comments on each filter below. The
    xG one matters for scope: it restricts this chart to league matches, because SofaScore
    doesn't model xG for cup competitions at all.'''

    # Prefer the team ID over the folder name: Shot_Map.csv splits the two sides by an isHome
    # flag only, so getting this backwards silently yields the OPPONENT's shots rather than
    # failing. Falls back to the name when the ID can't be resolved (a single-match `matches`,
    # or missing Match_Attributes), which is why `team` is still a parameter.
    team_id = resolve_team_id(matches)
    team_name = team.replace('_', ' ')

    matches_shots = []
    for match, match_files in matches.items():
        if 'Shot_Map.csv' not in match_files or 'Match_Attributes.csv' not in match_files:
            continue
        match_attributes = match_files['Match_Attributes.csv']

        if team_id is not None and 'homeTeam.id' in match_attributes.columns:
            is_home = int(match_attributes['homeTeam.id'].iloc[0]) == team_id
        else:
            is_home = match_attributes['homeTeam.name'].iloc[0] == team_name

        # `is_home != conceded` is an XOR: it keeps `team`'s own rows normally and flips to the
        # other side's when conceded=True, in both the home and away cases.
        match_shots = match_files['Shot_Map.csv'][match_files['Shot_Map.csv']['isHome'] == (is_home != conceded)].copy()
        # Match identity, for the same reason extract_corners carries it: corner_xg_created joins
        # deliveries to shots per fixture, and resolve_team_id deliberately returns None for a
        # single-match input, so re-deriving the split by calling this function per folder would
        # silently fall back to name matching.
        match_shots['Match'] = match
        matches_shots.append(match_shots)

    if not matches_shots:
        return pd.DataFrame(columns=SHOT_COLUMNS)

    shots = pd.concat(matches_shots).copy()

    # Own goals are filed under the BENEFITING team's isHome flag but carry the name of the
    # OPPOSITION player who scored them (48 such rows across MLS 2026) — so without this filter
    # a player key built from this data lists strangers from other clubs. Confirmed on real
    # data: Columbus Crew's shooter list gained Casemiro (Inter Miami) and Nathan Harriel
    # (Philadelphia Union), each with a single "shot". They also carry NaN xg/xgot, and are the
    # only rows in the file that do, so this filter and the finite-xG guard below overlap
    # completely — both are kept because they guard different things.
    if 'goalType' in shots.columns:
        shots = shots[shots['goalType'] != 'own']

    # SofaScore only models xG for LEAGUE competitions — measured across the whole data
    # directory, MLS and Liga MX carry it on 99.7% of shots while the CONCACAF Champions Cup,
    # Leagues Cup, US Open Cup, USL Championship and USL League One carry it on exactly ZERO
    # (0 of 2342, 3090, 3894, 5278 and 3268 shots respectively). Since the report is normally
    # run over ['MLS', 'Leagues_Cup', 'US Open Cup'], that silently removes a fifth to a third
    # of a team's shots — 120 of Charlotte FC's 613 and 158 of Columbus Crew's 549.
    #
    # They're dropped rather than plotted-without-a-value so that every number the chart shows
    # rests on the same set of shots: the heatmap, the npxG/90 ranking and the shot count would
    # otherwise disagree about which matches they cover. The print makes the gap visible rather
    # than leaving a reader to assume "25 shots" means the whole season.
    #
    # UNLESS THERE IS NO xG AT ALL, in which case dropping them would empty the frame and the
    # chart would simply not be produced. A team scouted purely on cup football has real shots
    # worth mapping — where they come from, who takes them, out of what situation — and only the
    # xG-derived layers have to go. The whole frame is kept with xg/xgot left NaN, has_xg()
    # reports the shortfall, and the charts degrade rather than disappear. The consistency
    # argument above is preserved either way: the frame is all-or-nothing on xG, never mixed.
    # xGOT for a shot that never reached the target is ZERO, and the two competitions record that
    # fact differently. MLS and Liga MX write a literal 0 on every miss, block and post; NCAA
    # leaves the cell blank. Reconstructing the 0 here rather than treating blank as "no data" is
    # what keeps a miss meaning the same thing in both — see the finishing note below.
    #
    # THIS IS WHY 62% OF NCAA SHOTS USED TO VANISH. Inclusion was gated on xg AND xgot both being
    # finite, so every off-target NCAA shot was dropped despite carrying a perfectly good xG:
    # 3,590 of 5,816 archive-wide, and 18 of Ohio State's 28. MLS and Liga MX lost nothing to it,
    # which is why it only ever showed up on a college side.
    off_target = shots['shotType'].isin(SHOT_OFF_TARGET_TYPES)
    shots.loc[off_target & ~np.isfinite(shots['xgot']), 'xgot'] = 0.0

    # Inclusion now rests on xg ALONE. That is the column the heatmap bins and the marker sizing
    # reads; xgot feeds only the xGOT-xG column, and a shot with no xgot is one row missing from
    # that sum rather than a shot that didn't happen. The remaining blanks are the handful of
    # ON-target shots with no recorded value (10 goals across all of NCAA), which cannot be
    # reconstructed and are skipped by the finishing sum's own NaN handling.
    missing_xg = ~np.isfinite(shots['xg'])
    described = f'shots conceded by {team}' if conceded else f'{team} shots'
    if missing_xg.all():
        if not quiet:
            print(f'None of the {len(shots)} {described} carry xG data (SofaScore does not model '
                  f'xG for cup or USL competitions) — the shot charts will drop their xG layers.')
    elif missing_xg.any():
        if not quiet:
            # Named per side, because a report page calls this for both directions and two
            # identically worded lines in the log read as a duplicate rather than as two different
            # numbers. quiet=True is for the second caller of the SAME direction — the shot map's
            # situation key needs the conceded frame for its counts, but the conceded chart reports
            # that exclusion already, and printing it twice makes a reader doubt the number.
            print(f'{missing_xg.sum()} of {len(shots)} {described} have no xG data '
                  f'(SofaScore does not model xG for cup or USL competitions) — excluded from the shot map.')
        shots = shots[~missing_xg]

    # See the docstring: both axes are flipped, not just scaled.
    shots['Shot X Coord'] = 100 - shots['playerCoordinates.x']
    shots['Shot Y Coord'] = 100 - shots['playerCoordinates.y']

    # Unrecognised situations fall back rather than being dropped — a shot is still a shot.
    shots['Situation Group'] = shots['situation'].map(SHOT_SITUATION_GROUPS).fillna(SHOT_SITUATION_DEFAULT)

    return shots

#%%
def matches_with_xg(matches):
    '''Subset of `matches` whose Shot_Map.csv actually carries xG.

    THE PER-90 DENOMINATORS DEPEND ON THIS. extract_shots drops every shot with no xG, which
    in practice means it drops entire competitions — SofaScore models xG for MLS and Liga MX
    and for nothing else (see extract_shots). Dividing that league-only xG by minutes played
    across ALL competitions understates every per-90 rate by however much cup football the team
    has played: 29% of Columbus Crew's shots and 20% of Charlotte FC's come from matches that
    contribute minutes but can never contribute xG.

    So every rate on the shot map uses these matches as its denominator, keeping numerator and
    denominator drawn from the same fixtures.'''
    usable = {}
    for folder, match_files in matches.items():
        shot_map = match_files.get('Shot_Map.csv')
        if shot_map is None or 'xg' not in shot_map.columns:
            continue
        if np.isfinite(shot_map['xg']).any():
            usable[folder] = match_files
    return usable

#%%
def ordinal(number):
    '''1 -> "1st", 2 -> "2nd", 11 -> "11th", 97 -> "97th".'''
    number = int(number)
    # 11/12/13 are the exception the last-digit rule gets wrong ("11st"), and they're reachable
    # here — a percentile runs 0-100.
    if number % 100 in (11, 12, 13):
        return f'{number}th'
    return f'{number}' + {1: 'st', 2: 'nd', 3: 'rd'}.get(number % 10, 'th')

#%%
def league_situation_xg_rates(leagues, team, team_matches, date):
    '''Per-situation xG per 90 for every team in the league, both directions, as a
    (for_rates, against_rates) pair of {team_id: {situation_group: xg_per_90}} dicts —
    (None, None) if they can't be built.

    Both come out of ONE pass, which is the reason they're returned together rather than behind a
    flag: load_league_data over a full season is by far the most expensive thing this module
    does, and every shot that counts toward the side that took it counts equally toward the side
    that faced it. Splitting this into two calls would double that cost to recompute a number the
    first pass already had in hand.

    The reference distribution behind the shot map's situation key. Built PER TEAM rather than
    as a single pooled average, because the key needs a percentile as well as an average, and a
    percentile needs the whole distribution — see the situation_label comments in
    generate_shot_map for why one number can't do both jobs.

    Every league match contributes both sides: a shot is a shot regardless of who took it, and
    both sides double the sample at no cost to correctness. Cup fixtures are skipped whole,
    since they carry no xG and would otherwise add minutes to a denominator they can never
    contribute to — the same trap matches_with_xg guards on the team side.

    Note that load_league_data's `exclude_team` is deliberately NOT used: Shot_Map.csv holds
    both teams' rows, so skipping a team's own folder still leaves their shots in every
    opponent's copy of the same fixture. Splitting by ID and isHome flag, as here, is what
    actually separates the teams — and it's what lets the caller both exclude the scouted team
    from the average and rank them against everyone else.'''

    league_matches = load_league_data(leagues, date, ['Shot_Map.csv'])

    team_xg, team_xg_against, team_minutes = {}, {}, {}
    for folder, match_files in league_matches.items():
        shot_map = match_files.get('Shot_Map.csv')
        match_attributes = match_files.get('Match_Attributes.csv')
        if shot_map is None or match_attributes is None:
            continue
        if 'isHome' not in shot_map.columns or 'homeTeam.id' not in match_attributes.columns:
            continue
        if 'xg' not in shot_map.columns or not np.isfinite(shot_map['xg']).any():
            continue

        if 'goalType' in shot_map.columns:
            shot_map = shot_map[shot_map['goalType'] != 'own']
        shot_map = shot_map[np.isfinite(shot_map['xg'])]
        if shot_map.empty:
            continue

        minutes = compute_total_minutes({folder: match_files})
        groups = shot_map['situation'].map(SHOT_SITUATION_GROUPS).fillna(SHOT_SITUATION_DEFAULT)
        is_home_flag = shot_map['isHome'].astype(bool)

        home_id = int(match_attributes['homeTeam.id'].iloc[0])
        away_id = int(match_attributes['awayTeam.id'].iloc[0])

        for side_id, opponent_id, side_rows in ((home_id, away_id, is_home_flag),
                                                (away_id, home_id, ~is_home_flag)):
            # Minutes accrue for both sides whether or not they managed a shot, so a team that
            # was shut out in a match still carries that match in its own denominator. The SAME
            # minutes are the denominator for both directions: a team's conceded rate is per 90
            # of their own football played, exactly as their scoring rate is.
            team_minutes[side_id] = team_minutes.get(side_id, 0) + minutes
            side_xg = team_xg.setdefault(side_id, {})
            # One shot, two ledgers — for whoever took it, against whoever faced it.
            opponent_xg = team_xg_against.setdefault(opponent_id, {})
            for group, total in shot_map[side_rows].groupby(groups[side_rows])['xg'].sum().items():
                side_xg[group] = side_xg.get(group, 0) + total
                opponent_xg[group] = opponent_xg.get(group, 0) + total

    def per_90(totals_by_team):
        return {
            side_id: {group: total / team_minutes[side_id] * 90 for group, total in totals.items()}
            for side_id, totals in totals_by_team.items() if team_minutes.get(side_id)
        }

    for_rates, against_rates = per_90(team_xg), per_90(team_xg_against)
    # Guarded together rather than separately: they come from the same pass, so if one is empty
    # the league data didn't load and neither number means anything.
    if not for_rates or not against_rates:
        return None, None
    return for_rates, against_rates

#%%
# bbox is square, unlike every other chart in this file. Cropping to the attacking half makes
# the visible pitch 52.5m long by 68m wide — TALLER than it is wide — and on an 8.5x11 page a
# figure-fraction width w renders at height w * (8.5/11) * (68/52.5) = 1.0008 * w. So a square
# bbox is exactly the shape that leaves no dead margin around an aspect-locked attacking half.
#
# 0.26 rather than the siblings' 0.28 because the height, not the width, is what binds here:
# the box has to sit between this chart's own notes box (top edge 0.17) and the notes box of
# the Key Passes chart above it (bottom edge 0.49), leaving room for a title above and an
# x-label below. 0.26 is the largest square that fits — and still renders the pitch 1.8x taller
# than the full-pitch version did (0.144 of page height, measured, vs 0.26 here).
def generate_shot_map(leagues, team, matches, date, team_colors, page,
                      bbox=[0.03, 0.20, 0.26, 0.26],
                      expected_lineup=None, expected_subs=None):
    '''Where the scouted team shoots from, who takes those shots, and out of what situation.

    Three encodings on one pitch: a ΣxG heatmap for the zones they generate danger from, dot
    COLOUR for the shooter, and dot SHAPE for the situation the shot came out of. The situation
    key carries each situation's xG per 90 against the league average, so the shapes answer
    "are they unusually dangerous from this?" rather than only "how often?".

    WITHOUT xG (a cup-only opponent — see has_xg) the chart keeps everything that is counted and
    drops everything that is modelled: no heatmap, no colourbar, uniform marker sizes, no size
    key, and the player and situation keys fall back to shot and goal counts. Shooters are then
    ranked by volume instead of npxG/90. What survives still answers where they shoot from, who
    takes them and out of what situation; only the danger dimension is gone, and the title says
    so rather than leaving a reader to wonder why this opponent's chart looks different.'''

    shots = extract_shots(matches, team)
    # Dropped once, here, so every number downstream agrees with the marks on the pitch — the
    # heatmap, the npxG/90 ranking, the per-player shot counts and the situation key all read
    # from `shots`. See SHOT_EXCLUDED_SITUATIONS for why penalties come out.
    shots = shots[~shots['Situation Group'].isin(SHOT_EXCLUDED_SITUATIONS)]

    if shots.empty:
        print('No shot data available — skipping shot map.')
        return page, None, None

    xg_available = has_xg(shots)
    # Set inside the heatmap branch below; stays None when there is no xG to scale.
    shared_vmax = None

    # The shots taken AGAINST this team, for the conceded half of each row. Free — Shot_Map.csv
    # holds both sides of every fixture, so this reads the frames already in `matches` (see
    # extract_shots). Penalties come out on the same terms as the offensive side: a spot kick
    # says as little about how a team defends a situation as it does about how they create one,
    # and leaving them in would put a flat 0.7884 on whichever row happened to catch them.
    # quiet=True: generate_def_shot_map extracts the same frame and reports any xG-less shots
    # itself, so letting this call print too would double the warning.
    conceded_shots = extract_shots(matches, team, conceded=True, quiet=True)
    conceded_shots = conceded_shots[~conceded_shots['Situation Group'].isin(SHOT_EXCLUDED_SITUATIONS)]
    # Hoisted above the heatmap, which needs it to put the pair on one colour scale. It is also
    # the source of the situation key's conceded columns further down.

    ax = page.add_axes(bbox)
    # pitch_color='none' so the pitch's own background matches the page instead of
    # painting a separate fill behind it — same as every other chart in this file.
    pitch = Pitch(pitch_type='opta', pitch_color='none', line_color='#888888', linewidth=1)
    pitch.draw(ax=ax)

    # Draw the FULL pitch (so the goal/box/centre-circle geometry stays correct) and then crop
    # to the shooting window, the same two-step generate_buildup_turnover_map uses to show only
    # the defensive half. See SHOT_MAP_X_MIN for what the window costs and buys.
    crop_shot_map(ax)

    # bin_statistic always spans the FULL pitch with no way to restrict its range, so the bin
    # count is chosen for what survives the crop: (12, 6) leaves 16 cells inside the window.
    #
    # Coarser than the (20, 10) this used to be, which was too fine to mean anything: measured on
    # Nashville's 226 shots it left 56 visible cells holding 4.0 shots each with 29% of them
    # empty, so the heatmap was largely reporting which cells happened to catch a shot. (12, 6)
    # holds 14.1 shots per cell with 6% empty. Still much finer than the (6, 5) the full-pitch
    # siblings use, because the shots are packed into a far smaller area — at (6, 5) the whole
    # penalty area fell inside a single cell.
    #
    # Skipped entirely without xG. A shot-COUNT heatmap was the obvious substitute and is
    # deliberately not drawn: the scatter already shows where the shots are, one mark per shot,
    # so binning the same rows again would restate it in a second visual language and imply the
    # colour carried something the marks didn't.
    mesh = None
    if xg_available:
        xg_bins = pitch.bin_statistic(shots['Shot X Coord'], shots['Shot Y Coord'],
                                        values=shots['xg'], statistic='sum', bins=SHOT_HEATMAP_BINS)

        cmap = LinearSegmentedColormap.from_list('cmap', ['#ffffff', team_colors[0]])
        # ONE scale across the pair, not this chart's own maximum. Self-normalising made a solid
        # cell here mean a different ΣxG from a solid cell on the conceded chart beside it, so the
        # two could not be read against each other — which is the first thing a reader does with a
        # pair. The conceded frame is binned unrotated on purpose: rotating maps cell (i,j) to
        # (N-i, M-j), so the SET of cell values, and therefore the maximum, is unchanged.
        conceded_bins = pitch.bin_statistic(conceded_shots['Shot X Coord'], conceded_shots['Shot Y Coord'],
                                            values=conceded_shots['xg'], statistic='sum',
                                            bins=SHOT_HEATMAP_BINS)
        max_val = np.nanmax([np.nanmax(xg_bins['statistic']), np.nanmax(conceded_bins['statistic'])])
        if not max_val or not np.isfinite(max_val):
            max_val = 1
        shared_vmax = max_val

        # Held back to a wash. At full strength the finer bins read as a hard mosaic that competes
        # with the marks in front of it, and its most saturated cell — always the one in front of
        # goal, where the marks are densest — swallowed the dots sitting on top of it. The heatmap
        # is context for the scatter here, not the subject.
        mesh = pitch.heatmap(xg_bins, ax=ax, cmap=cmap, vmin=0, vmax=max_val, zorder=1,
                             alpha=SHOT_HEATMAP_ALPHA)

        # Redraw the pitch lines on top of the heatmap fill so they're visible — same
        # zorder-bump fix used by every other heatmap chart in this file.
        existing_artists = set(ax.get_children())
        pitch.draw(ax=ax)
        crop_shot_map(ax)
        for artist in ax.get_children():
            if artist not in existing_artists:
                artist.set_zorder(mesh.get_zorder() + 1)

    # Minutes come from the xG-carrying matches only, NOT from `matches` — see matches_with_xg.
    # Using all of `matches` divided league-only xG by cup-inclusive minutes and understated
    # every per-90 rate on this chart by 20-30%.
    # Without xG there are no xG-carrying matches to narrow to, and narrowing anyway would leave
    # an empty denominator for every rate on the chart. The counts that replace those rates are
    # drawn from the same fixtures the marks are, so `matches` is the correct scope there.
    xg_matches = matches_with_xg(matches) if xg_available else matches
    player_minutes = compute_player_minutes(xg_matches)
    # Blocked shots are dropped from the finishing number, misses are NOT. xgot is recorded as
    # a literal 0 (not blank) for every miss, block and post — 242 of the 393 shots across both
    # sides of Columbus Crew's 2026 fixtures — so the scope of xGOT-xG is a real choice rather
    # than a formality; the zeros are counted or not counted, never skipped as missing.
    # A strike that misses the target is the shooter's own doing and should count against them;
    # a strike a defender gets a leg to is not, and blocks are ~23% of all shots, enough to
    # rank a high-volume box striker as a poor finisher purely for shooting through traffic.
    unblocked_shots = shots[shots['shotType'] != 'block']

    def non_penalty_xg_per_90(player):
        '''The ranking metric: how much open-play danger this player generates per 90.

        Non-penalty by construction — penalties are dropped from `shots` before this runs (see
        SHOT_EXCLUDED_SITUATIONS). That matters to the ranking, not just the chart: a designated
        taker banks 0.7884 xG per spot kick regardless of how threatening they are in open play,
        and counting them moved Dániel Gazdag (2 penalties) from 6th to 3rd on Columbus Crew's
        real list, ahead of players who out-threaten him from the run of play. Per 90 rather
        than a season total so a mid-season signing or a rotation player is ranked on their
        rate, not on how long they have been available.'''
        minutes = player_minutes.get(player, 0)
        total_xg = shots.loc[shots['player.name'] == player, 'xg'].sum()
        return total_xg / minutes * 90 if minutes else 0

    def finishing(player):
        '''ΣxGOT - ΣxG over unblocked shots: how much better (or worse) than the chance itself
        this player's actual strikes were.

        A shot whose xgot is still blank after extract_shots' reconstruction is one that WAS on
        target but carries no recorded value (10 goals in all of NCAA), and pandas' sum() skips
        the NaN it produces — so such a shot drops out of this total rather than voiding it. That
        is the right behaviour and it is relied on rather than incidental: there is no honest way
        to score the finishing of a strike whose on-target quality was never measured.'''
        player_shots = unblocked_shots[unblocked_shots['player.name'] == player]
        return (player_shots['xgot'] - player_shots['xg']).sum()

    def shot_count(player):
        '''The fallback ranking metric. Volume, not a rate: without xG there is nothing to
        normalise, and a per-90 shot count would rank a 20-minute substitute who had one effort
        above a striker with thirty.'''
        return int((shots['player.name'] == player).sum())

    rank_by = non_penalty_xg_per_90 if xg_available else shot_count
    players = sorted(shots['player.name'].dropna().unique(), key=rank_by, reverse=True)
    named_players, other_players = players[:SHOT_LEGEND_MAX], players[SHOT_LEGEND_MAX:]

    # Minutes the team itself played, as the denominator for the situation key's per-90 rates.
    # xg_matches, not matches, for the reason above.
    team_minutes = compute_total_minutes(xg_matches)
    qualified = player_qualifier(matches, expected_lineup, expected_subs, player_minutes)

    distinct_colors = distinct_player_colors()

    player_handles, player_values = [], []
    unqualified_labels = []

    for i, player in enumerate(named_players):
        color = distinct_colors[i % len(distinct_colors)]
        player_shots = shots[shots['player.name'] == player]

        plot_shot_marks(ax, player_shots, color, uniform_size=not xg_available)

        if xg_available:
            player_values.append((abbreviate_player_name(player), f'{non_penalty_xg_per_90(player):.2f}',
                                  len(player_shots), f'{finishing(player):+.2f}'))
        else:
            # Goals replace the two xG-derived numbers. It is the only outcome measure left that
            # is counted rather than modelled, and it keeps the row from collapsing to a bare
            # shot total that the ranking already conveys.
            goals = int((player_shots['shotType'] == 'goal').sum())
            player_values.append((abbreviate_player_name(player), len(player_shots), goals))
        player_handles.append(Patch(facecolor=color, edgecolor='black', linewidth=0.5))
        unqualified_labels.append(not qualified(player))

    # Everyone past the cap still gets their shots plotted — dropping them would remove real
    # events from the chart — but shares one grey swatch and a single legend row. They're the
    # least threatening shooters by construction, so the detail lost is the least interesting.
    if other_players:
        other_shots = shots[shots['player.name'].isin(other_players)]
        plot_shot_marks(ax, other_shots, '#999999', uniform_size=not xg_available)
        player_handles.append(Patch(facecolor='#999999', edgecolor='black', linewidth=0.5))
        player_values.append((f'Other ({len(other_players)})', '', len(other_shots), '')
                             if xg_available else (f'Other ({len(other_players)})',
                                                   len(other_shots),
                                                   int((other_shots['shotType'] == 'goal').sum())))

    pitch_pos = ax.get_position()
    # No heatmap means no colourbar — and the key column stays anchored to the axes, so the space
    # it frees simply goes unused rather than needing the whole layout rebuilt.
    if mesh is not None:
        cbar_ax = page.add_axes([pitch_pos.x1 + 0.02, pitch_pos.y0, 0.015, pitch_pos.height])
        cbar = page.colorbar(mesh, cax=cbar_ax)
        cbar.set_label('ΣxG / Zone (excl. penalties)', labelpad=6, fontsize=8)

    # Set before the legends below, which measure the title to align themselves against it.
    title_fontsize, title_pad = 10, 12
    ax.set_title('Shots Taken' if xg_available else f'Shots Taken ({SHOT_XG_FALLBACK_NOTE})',
                 fontsize=10, pad=title_pad)
    ax.set_xlabel('→    Attack    →', labelpad=8, fontsize=8)

    # Anchor the player key's TOP level with the title's top, rather than with the axes' top
    # edge (y=1.0) where the pitch ends — the title sits `pad` points above that. Converting
    # the pad plus ~0.8 * fontsize (to reach the top of the capitals) out of points and into
    # axes fractions keeps the two level even if the figure or font sizes change. Same
    # derivation as generate_key_pass_map; get_position() is the aspect-resolved box by now,
    # which is the assumption the colorbar placement above already relies on.
    axes_height_points = ax.get_position().height * page.get_figheight() * 72
    legend_top = 1.0 + (title_pad + 0.8 * title_fontsize) / axes_height_points

    # Colour swatches, not markers, for the player key — the shapes mean something else on this
    # chart, so giving each player a marker in their own key would imply a shooter is tied to a
    # situation. A flat rectangle reads unambiguously as "this colour is this player", and
    # contrasts with the situation key below it, which is drawn in markers and deliberately
    # colourless for the mirrored reason.
    # Grey rows fall short of KEY_PASS_MINUTES_QUALIFIED. Applied to the TEXT rather than the
    # swatch: the swatch colour is what ties a row to its dots on the pitch, so recolouring it
    # would break that link. The trailing "Other" row (when present) has no entry in
    # unqualified_labels, so zip stops before it.
    player_columns = ([('Player', 17, '<'), ('xG/90', 5, '>'), ('Sh', 4, '>'), ('±xGOT', 6, '>')]
                      if xg_available else
                      [('Player', 17, '<'), ('Sh', 4, '>'), ('G', 4, '>')])
    player_header, player_rows = format_key_table(player_columns, player_values)
    player_legend = table_legend(ax, player_header, player_rows, 'Goal-Scoring Threats',
                                 SHOT_LEGEND_ANCHOR_X, legend_top, handles=player_handles,
                                 grey=unqualified_labels, handlelength=1.0)

    # The situation key answers "which way do they hurt you?", so every number on it is measured
    # against the league rather than stated raw: every team creates most of its chances from
    # 'Assisted', so absolute shares say almost nothing about what makes THIS team dangerous.
    #
    # xG per 90 is the base quantity, not a share of the team's own shots and not a shot count.
    # It's the only one of the three that separates "they shoot from set pieces a lot" from
    # "their set-piece shots are dangerous", and Charlotte FC is the case that settles it: an
    # ordinary +0.4pp share of set-piece shots and a slightly BELOW-average 0.18 fewer of them
    # per 90, yet +0.17 xG/90 — the 97th percentile in MLS. Both count-based versions of this
    # number would have told a coach to ignore the best set-piece attack in the league. It also
    # keeps the graphic on one denomination: the heatmap is ΣxG, the player key npxG/90.
    #
    # Without xG the whole construction goes: the percentiles rank an xG/90 that doesn't exist,
    # and the league baseline they'd be ranked against is built from competitions this team never
    # played. The rows fall back to shot counts, which is the one thing still countable — and the
    # league pass is skipped rather than computed and discarded, since it is the single most
    # expensive call in this module.
    league_rates = league_rates_against = None
    if xg_available:
        # Both directions come back from one pass over the league — see league_situation_xg_rates.
        league_rates, league_rates_against = league_situation_xg_rates(leagues, team, matches, date)
    team_id = resolve_team_id(matches)
    other_team_rates = other_team_rates_against = None
    if league_rates:
        other_team_rates = [rates for side_id, rates in league_rates.items() if side_id != team_id]
        other_team_rates_against = [rates for side_id, rates in league_rates_against.items()
                                    if side_id != team_id]

    def situation_percentile(group, shot_rows, peer_rates_by_team):
        '''Where this team's xG per 90 from (or against) `group` ranks among the other teams.

        The percentile rather than the raw gap because each situation has its own spread and a
        raw gap isn't comparable down the column: measured across the 30 MLS teams, one standard
        deviation is 0.139 xG/90 for Assisted but 0.011 for Free Kick, so the same +0.17 is an
        ordinary 1.2 SD on the first row and a physically impossible 15 SD on the fifth. The
        percentile normalises each row to its own distribution, which is what lets a reader scan
        the column and find the outlier.'''
        team_rate = shot_rows.loc[shot_rows['Situation Group'] == group, 'xg'].sum() / team_minutes * 90
        peer_rates = [rates.get(group, 0) for rates in peer_rates_by_team]
        return np.mean([rate < team_rate for rate in peer_rates]) * 100

    def situation_label(group):
        '''"Set Piece  60, 38th | 54, 91st" — shots taken from this situation and where that
        threat ranks in the league, then shots faced from it and where THAT ranks.

        The two percentiles run in opposite directions on purpose, and the key's title says so.
        High on the left means they hurt you this way; high on the right means they are hurt this
        way, since it ranks xG conceded and conceding a lot of it is the bad end. A single row
        reading "38th | 91st" is the whole point of carrying both — an ordinary set-piece attack
        attached to a set-piece defence in the league's worst tenth is a game plan, and neither
        number alone would have found it.

        Counts are deliberately kept alongside each, because they are what say how much weight a
        percentile can bear — a rank built on 3 shots moves several places on one more, while the
        50-120 shot rows are stable.'''
        count = int((shots['Situation Group'] == group).sum())
        conceded_count = int((conceded_shots['Situation Group'] == group).sum())
        if not other_team_rates or not team_minutes:
            return (group, count, '', conceded_count, '')

        taken = ordinal(round(situation_percentile(group, shots, other_team_rates)))
        against = ordinal(round(situation_percentile(group, conceded_shots, other_team_rates_against)))
        return (group, count, taken, conceded_count, against)

    # Only situations that actually occur get a row — a team with no free kicks in the sample
    # shouldn't be given a dead legend entry implying zero is a finding. Grey fill rather than
    # any team or player colour, so the key reads as being about shape alone.
    #
    # A FIXED marker area here, not shot_marker_area: on the pitch size means xG, but in this
    # key each shape stands for a whole category, so sizing these by anything would imply the
    # category itself carries a magnitude.
    # A situation earns a row if it occurs in EITHER direction. The union rather than the taken
    # side alone because this key is also what decodes the shapes on the Shots Conceded chart
    # beside it — a situation this team never shoots from but regularly faces would otherwise
    # draw marks over there with nothing on the page explaining them.
    present_groups = set(shots['Situation Group']) | set(conceded_shots['Situation Group'])
    situation_handles = [
        Line2D([], [], linestyle='none', marker=marker, markerfacecolor='#999999',
               markeredgecolor='black', markeredgewidth=0.5,
               markersize=np.sqrt(SHOT_LEGEND_MARKER_AREA * SHOT_SHAPE_SIZE_FACTORS.get(group, 1.0)))
        for group, marker in SHOT_SITUATION_MARKERS.items() if group in present_groups
    ]
    situation_values = [situation_label(group) for group in SHOT_SITUATION_MARKERS
                        if group in present_groups]

    # What the marker's SIZE and its stripes mean. Built from empty scatters rather than Line2D
    # proxies because only a PathCollection carries a hatch, and the goal row has to show the
    # stripes it is explaining. Drawing no points, these add nothing to the pitch.
    #
    # Without xG the size rows go, because the marks are all one size — a key explaining a scale
    # the chart isn't using would be worse than no key. The GOAL row survives on its own, since
    # the heavier outline still means what it always did.
    goal_shape, goal_area_scale = goal_marker('o')
    scale_rows = [('o', 1.0, SHOT_MARKER_EDGE_WIDTH, value, f'{value:.2f} xG')
                  for value in SHOT_SIZE_LEGEND_VALUES] if xg_available else []
    scale_rows.append((goal_shape, goal_area_scale, goal_edge_width(),
                       0.30 if xg_available else SHOT_UNIFORM_XG, 'Goal'))
    scale_handles = [
        ax.scatter([], [], marker=shape, s=shot_marker_area(value, 'Assisted') * area_scale,
                   color='#999999', edgecolors='black', linewidths=edge_width,
                   alpha=SHOT_MARKER_ALPHA, label=label)
        for shape, area_scale, edge_width, value, label in scale_rows
    ]

    # Three keys stacked in ONE column, all left-aligned on the same x, each placed under the
    # measured bottom of the one above it. The block runs well below the bottom of the pitch,
    # which is fine here and only here: the notes box occupies x=0.03-0.29 while this column
    # starts around x=0.39, so the space below the pitch at this x is empty page.
    #
    # Measured rather than hardcoded because each key's height varies with the data — the player
    # key alone swings between 13 and 21 rows depending on how many shooters a squad has, and
    # the situation key drops rows for situations a team never used. A draw is needed first:
    # legends have no window extent until something has laid them out. The fallbacks are sized
    # for a full-height key, so they err toward too much gap rather than an overlap.
    # The two percentile columns run in OPPOSITE directions and the headers are what say so:
    # high under Take means they hurt you that way, high under Conc means they get hurt that way,
    # since it ranks xG conceded. As a table the pairing is visible down the column, which the
    # old "60, 38th | 54, 91st" run-on made a reader parse row by row.
    situation_columns = ([('Situation', 10, '<'), ('Sh', 5, '>'), ('Pctl', 5, '>'),
                          ('Sh', 5, '>'), ('Pctl', 5, '>')] if xg_available else
                         [('Situation', 10, '<'), ('Sh', 5, '>'), ('', 0, '>'),
                          ('Sh', 5, '>'), ('', 0, '>')])
    # A grouping row above the column names, so the two "Sh" columns are unambiguous.
    group_header, _ = format_key_table(
        [('', 10, '<'), ('Taken', 10, '^'), ('Conceded', 10, '^')], [])

    # The percentile column is headed "xG/90 Pctl", STACKED across two rows rather than written
    # on one. It has to be: "xG/90 Pctl" is 11 characters, and carrying it inline on both
    # percentile columns takes the table from 30 characters to 42 — against a page-three budget
    # of 33 between the colourbar label at x=0.408 and the conceded chart's pitch at x=0.600.
    # Centring "xG/90" in the same 5-wide column that holds "Pctl" puts it directly above, which
    # reads as the one heading it is and costs nothing.
    header_rows = [group_header]
    if xg_available:
        metric_header, _ = format_key_table(
            [('', 10, '<'), ('', 5, '>'), ('xG/90', 5, '^'), ('', 5, '>'), ('xG/90', 5, '^')], [])
        header_rows.append(metric_header)
    situation_header, situation_rows = format_key_table(situation_columns, situation_values)
    header_rows.append(situation_header)

    situation_top = legend_bottom_in_axes(page, ax, player_legend, -0.05) - SHOT_LEGEND_GAP
    situation_legend = table_legend(ax, header_rows, situation_rows, 'Shot Situations',
                                    SHOT_LEGEND_ANCHOR_X, situation_top,
                                    handles=situation_handles, handlelength=1.0,
                                    labelspacing=KEY_TABLE_MARKER_SPACING)

    # Not a table — one column of labelled marks, where the mark IS the content. labelspacing is
    # looser than the tables': these rows carry markers of deliberately different sizes, and at
    # 0.3 the largest one overlaps the row above it.
    scale_top = legend_bottom_in_axes(page, ax, situation_legend, -0.55) - SHOT_LEGEND_GAP
    stacked_legend(ax, scale_handles,
                   'Marker Size = Shot xG' if xg_available else 'Marker Outline',
                   SHOT_LEGEND_ANCHOR_X, scale_top, labelspacing=0.8, handlelength=1.0)

    # Aligned to the chart's own x-extent rather than centred on a page half — at bbox[0]=0.03
    # the page-half centre would put the box well to the right of the pitch it belongs to, and
    # the NOTES_WIDTH_PAD overhang would push its left edge outside the page's inner border.
    #
    # Unlike the siblings, the y is NOT derived from bbox[1]: this chart's bbox is sized to the
    # cropped pitch exactly (see the note above the signature), so there is no aspect-lock slack
    # between bbox[1] and where the pitch really ends, and the siblings' bbox[1]+0.05-0.06
    # offset would put the box straight through the x-label. Fixed instead, leaving 0.03 of
    # clearance below the pitch for that label.
    notes_bbox = draw_notes_box(page, [bbox[0], SHOT_NOTES_Y0, NOTES_WIDTH,
                                       bbox[1] - 0.03 - SHOT_NOTES_Y0])

    return page, notes_bbox, shared_vmax

#%%
# Same square bbox as generate_shot_map, at the same y and on the right half of the page —
# completing the 2x2 grid page three already half-builds (Key Passes / Key Passes Conceded above,
# Shots Taken / Shots Conceded here). The square is right for the same reason it is there: the
# mirrored crop window is the reflection of the offensive one, so it is the same 35 x-units by
# 70 y-units and resolves to the same aspect.
#
# x=0.60 is the same left edge generate_def_key_pass_map uses, and it clears the shot map's key
# column — but not by much, and the margin is the thing to watch if this chart ever moves. That
# column is anchored at axes x=1.40 (figure x=0.394) and its width is set by its single longest
# row: measured on Nashville SC it runs to 0.578 on "R. Baker-Whiting (0.08, 4, -0.40)", leaving
# 0.022 of clearance. It is bounded there only by abbreviate_player_name, which exists precisely
# because the unabbreviated key-pass list ran to 0.613 and collided with the conceded chart above
# — the same 0.600 edge, one row up. A squad with a longer compound surname is what would break
# this, and the fix is that function, not this bbox.
def generate_def_shot_map(team, matches, team_colors, page, bbox=[0.60, 0.20, 0.26, 0.26],
                          shared_vmax=None):
    '''Where the scouted team CONCEDES shots — the defensive mirror of generate_shot_map.

    Encodes the same three things as its offensive sibling: a ΣxG heatmap for the zones they are
    hurt from, marker SHAPE for the situation the shot came out of, and marker SIZE for the
    chance's xG, with goals set apart by a heavy outline.

    What it drops is the player dimension. The shots come from whichever opponents the team
    happened to face, so a per-player key would list strangers from a dozen unrelated clubs
    rather than a squad — the same reasoning that leaves generate_def_key_pass_map without one,
    and the reason every mark here is a single flat team_colors[1] over a team_colors[0] heatmap.

    It carries NO keys of its own. The situation shapes and the size scale are already spelled
    out by the Shots Taken keys to its left, and repeating them would spend a third of the page
    restating what a reader can see two inches away. The cost is that this chart's per-situation
    counts and league percentiles go unshown — deliberate, not an oversight.'''

    # Both sides of every fixture live in Shot_Map.csv, so this needs no loading the shot map
    # hasn't already done — see extract_shots for why conceded=True is the whole difference.
    shots = extract_shots(matches, team, conceded=True)
    # Penalties come out here for the same reason they come out of the offensive chart: every one
    # sits at exactly the same point at a flat 0.7884 xG, so a season's worth stacks the chart's
    # largest markers on its single most crowded pixel. See SHOT_EXCLUDED_SITUATIONS.
    shots = shots[~shots['Situation Group'].isin(SHOT_EXCLUDED_SITUATIONS)]

    if shots.empty:
        print('No conceded shot data available — skipping conceded shot map.')
        return page, None

    shots = shots.copy()

    # Rotate into the SCOUTED team's frame: their own goal at x=0, the opposition attacking
    # leftward toward it. extract_shots returns coordinates in the SHOOTER's frame (attacking
    # toward x=100), which for a conceded shot is an opponent's frame, not this team's.
    #
    # BOTH axes flip — see rotate_into_opponent_frame. These coordinates come from Shot_Map.csv
    # rather than the event file, but the same rotation applies: extract_shots normalises them
    # into the shooter's own frame, and a key-pass corner's end coordinate matches its resulting
    # shot at a median distance of 0.0 within a team, which is what confirms the two files share
    # one convention.
    shots = rotate_into_opponent_frame(shots, [('Shot X Coord', 'Shot Y Coord')])

    xg_available = has_xg(shots)

    ax = page.add_axes(bbox)
    # pitch_color='none' so the pitch's own background matches the page instead of
    # painting a separate fill behind it — same as every other chart in this file.
    pitch = Pitch(pitch_type='opta', pitch_color='none', line_color='#888888', linewidth=1)
    pitch.draw(ax=ax)

    # Draw the FULL pitch, then crop — same two-step as the offensive sibling. mirrored=True is
    # what makes the window the reflection of that chart's, around the goal at x=0.
    crop_shot_map(ax, mirrored=True)

    # Same bins as the offensive sibling, so a zone here is the same patch of grass as a zone
    # there — which, together with the shared scale below, is what makes the pair comparable.
    # Skipped without xG, exactly as on the offensive sibling — see the note there for why a
    # shot-COUNT heatmap is not substituted in its place.
    mesh = None
    if xg_available:
        xg_bins = pitch.bin_statistic(shots['Shot X Coord'], shots['Shot Y Coord'],
                                        values=shots['xg'], statistic='sum', bins=SHOT_HEATMAP_BINS)

        cmap = LinearSegmentedColormap.from_list('cmap', ['#ffffff', team_colors[0]])
        # shared_vmax comes from the offensive sibling, which sees both directions' data and so is
        # the only one of the pair able to compute a maximum across them. Falling back to this
        # chart's own maximum only happens when that chart was skipped entirely.
        max_val = shared_vmax if shared_vmax else np.nanmax(xg_bins['statistic'])
        if not max_val or not np.isfinite(max_val):
            max_val = 1

        # Held back to a wash, as on the offensive chart — the heatmap is context for the marks
        # here, not the subject, and at full strength its hottest cell (always the one in front of
        # goal, where the marks are densest) swallows the marks sitting on top of it.
        mesh = pitch.heatmap(xg_bins, ax=ax, cmap=cmap, vmin=0, vmax=max_val, zorder=1,
                             alpha=SHOT_HEATMAP_ALPHA)

        # Redraw the pitch lines on top of the heatmap fill so they're visible — same
        # zorder-bump fix used by every other heatmap chart in this file. pitch.draw resets the
        # limits, so the crop has to be reapplied after it.
        existing_artists = set(ax.get_children())
        pitch.draw(ax=ax)
        crop_shot_map(ax, mirrored=True)
        for artist in ax.get_children():
            if artist not in existing_artists:
                artist.set_zorder(mesh.get_zorder() + 1)

    # One call for the lot, where the offensive chart makes one per shooter — there is no player
    # dimension to split on. See DEF_SHOT_MARKER_ALPHA for why the opacity drops.
    plot_shot_marks(ax, shots, team_colors[1], alpha=DEF_SHOT_MARKER_ALPHA,
                    uniform_size=not xg_available)

    pitch_pos = ax.get_position()
    if mesh is not None:
        cbar_ax = page.add_axes([pitch_pos.x1 + 0.02, pitch_pos.y0, 0.015, pitch_pos.height])
        cbar = page.colorbar(mesh, cax=cbar_ax)
        cbar.set_label('ΣxG Conceded / Zone (excl. penalties)', labelpad=6, fontsize=8)

    ax.set_title('Shots Conceded' if xg_available else f'Shots Conceded ({SHOT_XG_FALLBACK_NOTE})',
                 fontsize=10, pad=12)
    # Arrow points LEFT because the opposition attacks toward x=0 in this mirrored frame — the
    # visible cue that this is a defensive mirror, matching the chart directly above it.
    ax.set_xlabel('←    Opposition Attack    ←', labelpad=8, fontsize=8)

    # Centred on the right half of the page like generate_def_key_pass_map, which at this width
    # puts the left edge at 0.60 — flush with the chart's own, since the NOTES_WIDTH_PAD overhang
    # happens to cancel the offset here.
    #
    # The y is taken from generate_shot_map's derivation rather than the siblings' bbox[1]+0.05
    # -0.06: this chart's bbox is sized to the cropped pitch exactly, so there is no aspect-lock
    # slack to absorb that offset and it would put the box straight through the x-label.
    notes_x0 = 0.75 - NOTES_WIDTH / 2
    notes_bbox = draw_notes_box(page, [notes_x0, SHOT_NOTES_Y0, NOTES_WIDTH,
                                       bbox[1] - 0.03 - SHOT_NOTES_Y0])

    return page, notes_bbox

#%%
# Where page four's key column lives, in the LEFT chart's axes fractions. Its bbox is only 0.208
# wide (the corner window is portrait — see CORNER_MAP_X_MIN), so the same axes fraction lands
# much further left here than the shot map's does, and the column collided with the colourbar's
# rotated label at x=0.335. 2.00 puts the left edge at 0.03 + 2.00*0.208 = 0.446, clear of it.
CORNER_LEGEND_ANCHOR_X = 1.56

def corner_shooter_rows(matches, team, expected_lineup=None, expected_subs=None):
    '''(header, rows) for the "Corner Shot Threat" table: who gets on the end of this team's
    corners, on the same three measures the shot map's player key uses.

    Scoped to every shot with situation == 'corner', NOT to the first-contact subset the heatmap
    is built from — a knockdown hammered in from the edge of the box is still their corner
    routine producing a chance, and on Nashville roughly half the corner xG arrives that way.
    The two numbers therefore differ by design; the colourbar says "first contact" so a reader
    isn't left to reconcile them.

    The rate column is headed xG/90, not npxG/90 as on the shot map. The two are the same number
    here — a penalty is never filed under situation == 'corner' — and calling it "np" would imply
    a distinction this scope cannot contain. Minutes come from the xG-carrying matches, the same
    denominator the shot map uses, so the rates on the two pages are comparable.

    Headers stay a COUNT while everything beside them is a rate. bodyPart is recorded (20 head /
    19 right / 7 left on Nashville) and aerial threat is the question a corner defence sets up to
    answer, but the per-90 of a 0-3 event count is noise dressed as precision — 3 headers in 1500
    minutes renders as 0.18, and a reader cannot tell that from 0.12 on two headers.'''
    shots = extract_shots(matches, team, quiet=True)
    shots = shots[shots['situation'] == 'corner']
    if shots.empty:
        return None, [], [], shots

    # bodyPart is present on every real Shot_Map, but guarded so a thin or malformed file degrades
    # to a zero column rather than taking the whole page down.
    if 'bodyPart' not in shots.columns:
        shots = shots.assign(bodyPart='')

    xg_available = has_xg(shots)
    player_minutes = compute_player_minutes(matches_with_xg(matches) if xg_available else matches)
    # Blocked shots come out of the finishing number for the reason generate_shot_map spells out:
    # a strike a defender gets a leg to is not the shooter's doing, and xgot is a literal 0 on
    # every block, so leaving them in would read as a finishing failure.
    unblocked = shots[shots['shotType'] != 'block']

    summary = shots.groupby('player.name').agg(
        shots=('xg', 'size'), xg=('xg', 'sum'),
        headers=('bodyPart', lambda column: int((column == 'head').sum())),
    )
    summary['minutes'] = [player_minutes.get(player, 0) for player in summary.index]
    summary['xg90'] = np.where(summary['minutes'] > 0,
                               summary['xg'] / summary['minutes'].replace(0, np.nan) * 90, 0)
    summary['finishing'] = [(unblocked.loc[unblocked['player.name'] == player, 'xgot']
                             - unblocked.loc[unblocked['player.name'] == player, 'xg']).sum()
                            for player in summary.index]

    # Without xG the two xG-derived columns go and the table ranks by volume instead — the same
    # substitution the shot map's player key makes, so the two read consistently on a cup-only
    # opponent.
    summary = summary.sort_values('xg90' if xg_available else 'shots', ascending=False)
    summary = summary.head(CORNER_LIST_MAX)

    if xg_available:
        # THREE decimals, where every other rate in the report uses two. Corner xG per 90 is an
        # order of magnitude smaller than the shot map's npxG/90 — it spans 0.008 to 0.082 on
        # Nashville against 0.04 to 0.55 there — so at two decimals seven of the ten rows printed
        # an identical "0.02" and the column ranked nothing.
        columns = [('Player', 17, '<'), ('xG/90', 6, '>'), ('Sh', 4, '>'),
                   ('±xGOT', 6, '>'), ('H', 3, '>')]
        values = [(abbreviate_player_name(p), f'{r.xg90:.3f}', int(r.shots),
                   f'{r.finishing:+.2f}', int(r.headers)) for p, r in summary.iterrows()]
    else:
        columns = [('Player', 17, '<'), ('Sh', 4, '>'), ('H', 3, '>')]
        values = [(abbreviate_player_name(p), int(r.shots), int(r.headers))
                  for p, r in summary.iterrows()]

    header, rows = format_key_table(columns, values)
    # Greyed on the same rule as every other player key — a corner threat who has banked two
    # cameos should not read like one who plays every week. See player_qualifier.
    qualified = player_qualifier(matches, expected_lineup, expected_subs, player_minutes)
    grey = [not qualified(player) for player in summary.index]
    return header, rows, grey, shots

# Portrait rather than the shot maps' square, because this chart keeps the full pitch WIDTH — see
# CORNER_MAP_X_MIN. The visible pitch is 40 x-units by 100 y-units, i.e. 42m by 68m, which
# aspect-locks a 0.26-tall bbox to 0.208 wide. Deriving the width from the height rather than the
# other way round is deliberate: the height is what's fixed, by the band between the page header
# at y=0.820 and the row of notes boxes.
#
# bbox[1]=0.525 puts the chart's top at 0.785, which is what lands its title in the same band
# page three's top row uses (measured: those titles span y=0.791-0.813). generatePageTemplate
# draws the header separator at y=0.820, and a title placed any higher runs into it.
def generate_corner_map(leagues, team, matches, date, team_colors, page,
                        bbox=[0.03, 0.525, 0.208, 0.26],
                        expected_lineup=None, expected_subs=None):
    '''Where this team's corners are delivered, who takes them, and which ones become chances.

    Four encodings: a heatmap of the xG those deliveries created, mark COLOUR for the taker,
    mark SHAPE for which arc it was struck from, and a HOLLOW mark for a delivery the defence
    cleared. Beside it, a text list of who actually shoots from their corners.

    Colour is the taker rather than the receiver because the receiver is not recoverable from
    this data at all — Player_Event_Data.csv is grouped by player, carries no timestamp and holds
    one team's events, so there is no next touch to read. The taker is the better encoding
    regardless: it is known for every delivery, it is concentrated enough to read (two players
    took 77% of Nashville's), and it pairs with the arc shape to show a routine. "Who is
    dangerous" is answered by the shooter list, over all 46 corner shots rather than the 25 the
    colour channel could have reached.

    Returns (page, notes_bbox, key_column), where key_column is an (axes, bottom_figure_y) handle
    the conceded chart hangs its own list from so the two stacks share one column.'''

    corners = extract_corners(matches)
    if not corners.empty:
        valid_end = np.isfinite(corners['Pass End X Coord']) & np.isfinite(corners['Pass End Y Coord'])
        corners = corners[valid_end].copy()

    if corners.empty:
        print('No corner data available — skipping corner map.')
        return page, None, None, None

    xg_created = corner_xg_created(corners, matches, matches, team)

    # Expected threat of the spot each ball was PUT, which is what the Taker key's quality column
    # averages. The same lookup and the same orientation generate_key_pass_map uses for its own
    # xT columns — these coordinates are in the taker's own frame, still attacking toward x=100.
    #
    # xT rather than the xG each corner actually created: only 21% of corners produce a shot at
    # all, so a per-taker mean of xG created is mostly an average of zeros and separates nobody.
    # Every corner has an xT, and across Nashville's takers it spans 0.072 to 0.258 — a 3.6x
    # spread that says who is putting the ball somewhere dangerous.
    corners['xT Delivered'] = lookup_xt(corners['Pass End X Coord'].to_numpy(),
                                        corners['Pass End Y Coord'].to_numpy(),
                                        attacking_toward_100=True)
    # The heatmap and the shooter list's xG column both rest on Shot_Map's xG, so they stand or
    # fall together with it — see has_xg and the fallback note in generate_shot_map.
    xg_available = has_xg(extract_shots(matches, team, quiet=True))
    # Set inside the heatmap branch below; stays None when there is no xG to scale.
    shared_vmax = None

    # The conceded deliveries, for the Delivery key's second count column only — nothing on this
    # chart is drawn from them. That key sits in the column BETWEEN the two charts and decodes
    # both, so a single set of counts beside it would read as applying to whichever chart the
    # reader was looking at, and the two totals genuinely differ (112 taken, 107 conceded on
    # Nashville). load_actual_opponent_events is cached in-process, so the conceded chart calling
    # it again costs nothing.
    opponent_matches = load_actual_opponent_events(matches, leagues, date, team)
    conceded_corners = extract_corners(opponent_matches)
    if not conceded_corners.empty:
        valid = (np.isfinite(conceded_corners['Pass End X Coord'])
                 & np.isfinite(conceded_corners['Pass End Y Coord']))
        conceded_corners = conceded_corners[valid]
    if not conceded_corners.empty:
        # Same re-derivation the conceded chart does, so its counts describe the shapes as they
        # are drawn over there rather than in the delivering team's frame. See
        # generate_def_corner_map.
        conceded_corners = conceded_corners.copy()
        conceded_corners['Arc'] = ~conceded_corners['Arc']

    ax = page.add_axes(bbox)
    # pitch_color='none' so the pitch's own background matches the page instead of
    # painting a separate fill behind it — same as every other chart in this file.
    pitch = Pitch(pitch_type='opta', pitch_color='none', line_color='#888888', linewidth=1)
    pitch.draw(ax=ax)
    crop_corner_map(ax)

    # Binned by where the delivery ENDED, valued by the xG it created — the same
    # bin-by-one/value-by-another construction generate_key_pass_map uses, and the thing that
    # separates "they put a lot of balls here" from "the balls they put here hurt you". A count
    # heatmap would simply light up the centre of the six-yard box on every team in the league.
    #
    # Coarser than the shot maps' grid — see CORNER_HEATMAP_BINS for what the old one was doing.
    mesh = None
    if xg_available:
        xg_bins = pitch.bin_statistic(corners['Pass End X Coord'], corners['Pass End Y Coord'],
                                      values=xg_created, statistic='sum', bins=CORNER_HEATMAP_BINS)

        cmap = LinearSegmentedColormap.from_list('cmap', ['#ffffff', team_colors[0]])
        # ONE scale across the pair, computed here because this is the only one of the two charts
        # that holds both directions' data. Self-normalising made a solid cell here mean a
        # different ΣxG from a solid cell on the conceded chart beside it, so the two could not be
        # read against each other. The conceded deliveries are binned UNROTATED on purpose:
        # rotating maps cell (i,j) to (N-i, M-j), so the set of cell values — and therefore the
        # maximum — is unchanged.
        conceded_created = corner_xg_created(conceded_corners, opponent_matches, matches,
                                             team, conceded=True)
        conceded_bins = pitch.bin_statistic(conceded_corners['Pass End X Coord'],
                                            conceded_corners['Pass End Y Coord'],
                                            values=conceded_created, statistic='sum',
                                            bins=CORNER_HEATMAP_BINS)
        max_val = np.nanmax([np.nanmax(xg_bins['statistic']), np.nanmax(conceded_bins['statistic'])])
        if not max_val or not np.isfinite(max_val):
            max_val = 1
        shared_vmax = max_val

        mesh = pitch.heatmap(xg_bins, ax=ax, cmap=cmap, vmin=0, vmax=max_val, zorder=1,
                             alpha=SHOT_HEATMAP_ALPHA)

        # Redraw the pitch lines on top of the heatmap fill so they're visible — same
        # zorder-bump fix used by every other heatmap chart in this file. pitch.draw resets the
        # limits, so the crop has to be reapplied after it.
        existing_artists = set(ax.get_children())
        pitch.draw(ax=ax)
        crop_corner_map(ax)
        _ = None
        for artist in ax.get_children():
            if artist not in existing_artists:
                artist.set_zorder(mesh.get_zorder() + 1)

    takers = corners['Player'].value_counts()
    named_takers = list(takers.head(CORNER_TAKER_LEGEND_MAX).index)
    other_takers = list(takers.index[CORNER_TAKER_LEGEND_MAX:])

    distinct_colors = distinct_player_colors()
    taker_qualified = player_qualifier(matches, expected_lineup, expected_subs)
    taker_handles, taker_values, taker_grey = [], [], []
    for i, taker in enumerate(named_takers):
        color = distinct_colors[i % len(distinct_colors)]
        taker_corners = corners[corners['Player'] == taker]
        plot_corner_marks(ax, taker_corners, color)
        taker_handles.append(Patch(facecolor=color, edgecolor='black', linewidth=0.5))
        taker_values.append((abbreviate_player_name(taker), len(taker_corners),
                             f'{taker_corners["Completed"].mean() * 100:.0f}%',
                             f'{taker_corners["xT Delivered"].mean():.3f}'))
        taker_grey.append(not taker_qualified(taker))

    # Everyone past the cap is still plotted — dropping them would remove real deliveries — but
    # shares one grey swatch, exactly as the shot map's tail does.
    if other_takers:
        other_corners = corners[corners['Player'].isin(other_takers)]
        plot_corner_marks(ax, other_corners, '#999999')
        taker_handles.append(Patch(facecolor='#999999', edgecolor='black', linewidth=0.5))
        taker_values.append((f'Other ({len(other_takers)})', len(other_corners),
                             f'{other_corners["Completed"].mean() * 100:.0f}%',
                             f'{other_corners["xT Delivered"].mean():.3f}'))

    pitch_pos = ax.get_position()
    if mesh is not None:
        cbar_ax = page.add_axes([pitch_pos.x1 + 0.02, pitch_pos.y0, 0.015, pitch_pos.height])
        cbar = page.colorbar(mesh, cax=cbar_ax)
        # "first contact" is load-bearing, not a hedge — see corner_xg_created for what it excludes.
        cbar.set_label('ΣxG Created / Zone (first contact)', labelpad=6, fontsize=8)

    title_fontsize, title_pad = 10, 12
    ax.set_title('Corners Taken' if xg_available else f'Corners Taken ({SHOT_XG_FALLBACK_NOTE})',
                 fontsize=10, pad=title_pad)
    ax.set_xlabel('→    Attack    →', labelpad=8, fontsize=8)

    # Anchor the first key's top to the title's top rather than the axes' top edge — same
    # derivation as generate_shot_map's.
    axes_height_points = ax.get_position().height * page.get_figheight() * 72
    legend_top = 1.0 + (title_pad + 0.8 * title_fontsize) / axes_height_points

    taker_header, taker_rows = format_key_table(
        [('Taker', 17, '<'), ('Corners', 8, '>'), ('%Cmp', 6, '>'), ('xT/Cnr', 7, '>')], taker_values)
    taker_legend = table_legend(ax, taker_header, taker_rows, 'Corner Takers',
                                CORNER_LEGEND_ANCHOR_X, legend_top,
                                handles=taker_handles, grey=taker_grey, handlelength=1.0)

    # Arc and outcome in ONE key rather than two. They are separate encodings, but a reader meets
    # them as four combinations of mark, and tabulating exactly the four that occur — with counts
    # both ways — costs one title instead of two and cannot describe a combination the chart
    # doesn't contain.
    def combination_count(frame, arc, completed):
        if frame.empty:
            return 0
        return int(((frame['Arc'] == arc) & (frame['Completed'] == completed)).sum())

    delivery_handles, delivery_values = [], []
    for arc, marker in CORNER_ARC_MARKERS.items():
        for completed in (True, False):
            taken_count = combination_count(corners, arc, completed)
            conceded_count = combination_count(conceded_corners, arc, completed)
            if not taken_count and not conceded_count:
                continue
            delivery_handles.append(Line2D(
                [], [], linestyle='none', marker=marker,
                markerfacecolor='#999999' if completed else 'none',
                markeredgecolor='black' if completed else '#999999',
                markeredgewidth=0.5 if completed else CORNER_HOLLOW_EDGE_WIDTH,
                markersize=np.sqrt(SHOT_LEGEND_MARKER_AREA)))
            delivery_values.append((f'{CORNER_ARC_LABELS[arc]}, {"completed" if completed else "cleared"}',
                                    taken_count, conceded_count))

    delivery_header, delivery_rows = format_key_table(
        [('Delivery', 22, '<'), ('Taken', 6, '>'), ('Conc', 6, '>')], delivery_values)
    delivery_top = legend_bottom_in_axes(page, ax, taker_legend, -0.10) - SHOT_LEGEND_GAP
    delivery_legend = table_legend(ax, delivery_header, delivery_rows, 'Corner Deliveries',
                                   CORNER_LEGEND_ANCHOR_X, delivery_top,
                                   handles=delivery_handles, handlelength=1.0,
                                   labelspacing=KEY_TABLE_MARKER_SPACING)

    shooter_header, shooter_rows, shooter_grey, _ = corner_shooter_rows(
        matches, team, expected_lineup, expected_subs)
    key_bottom = delivery_top
    if shooter_rows:
        shooter_top = legend_bottom_in_axes(page, ax, delivery_legend, -0.45) - SHOT_LEGEND_GAP
        shooter_legend = table_legend(ax, shooter_header, shooter_rows, 'Corner Shot Threat',
                                      CORNER_LEGEND_ANCHOR_X, shooter_top, grey=shooter_grey)
        key_bottom = legend_bottom_in_axes(page, ax, shooter_legend, -0.80)

    # Converted out of this chart's axes fractions into figure fractions, because the conceded
    # chart hangs its own list from here and lives in a different (equally sized, but offset)
    # axes.
    key_bottom_figure = pitch_pos.y0 + key_bottom * pitch_pos.height

    # Aligned to the chart's own x-extent rather than centred on a page half, and the y is fixed
    # rather than derived from bbox[1] — both for the reasons generate_shot_map gives: the bbox
    # is sized to the cropped pitch exactly, so there is no aspect-lock slack to absorb an offset.
    notes_bbox = draw_notes_box(page, [bbox[0], CORNER_NOTES_Y0, NOTES_WIDTH,
                                       bbox[1] - 0.03 - CORNER_NOTES_Y0])

    return page, notes_bbox, (ax, key_bottom_figure), shared_vmax

#%%
def corner_defender_rows(counts, conceded_corners, matches,
                         expected_lineup=None, expected_subs=None):
    '''(header, rows, grey) for the "Corners Defended" table: who deals with the corners this
    team faces.

    Two measures that answer different questions, which is the point of carrying both:

      Act/90  - how involved this player is per minute on the pitch. A rate.
      Act     - how many they actually won. The raw volume behind that rate.

    The pair separates a high rate on thin evidence from real workload, and the table is sorted by
    the first of them. On Nashville the backup keeper Xavier Valdez posts the highest rate on the
    squad (2.50/90) off 180 minutes but only 9% of the team's corner defending, while Brian
    Schwake sits at 0.71/90 and 27% — so the two columns disagree about the order on purpose, and
    a reader comparing them is reading the point of the table rather than an inconsistency.

    The raw count REPLACES a "%Team" share of the team's matched actions, which replaced an
    earlier "% of corners faced". That first one divided actions by the corners the player was
    estimated to be on the pitch for, scaling the team's corners faced by the player's share of
    team minutes — and the minutes cancelled, making it exactly Act/90 times a constant (23.904 on
    Nashville, identical for every player). %Team was correct but bespoke, and it is the same
    count rescaled, so the plain number says the same thing and matches the (rate, volume) shape
    the other three player keys use. Counting corners faced per player for real is not possible
    here: corners carry no timestamp anywhere in this data, so there is nothing to intersect
    against a substitution.

    Rows below the minutes threshold are greyed rather than dropped, the device the shot map's
    player key already uses — Valdez's 2.50/90 is real but rests on two matches, and the grey is
    what says so without removing a player who might start.

    Goalkeepers are deliberately not filtered out. On Nashville the two keepers account for 20 of
    56 actions, 14 of them ball-recoveries — which at a corner is a claim, and "their keeper comes
    for everything" is one of the more useful things a corner routine can know.'''
    if not counts:
        return None, [], []

    player_minutes = compute_player_minutes(matches)
    qualified = player_qualifier(matches, expected_lineup, expected_subs, player_minutes)

    # Ranked by Act/90, the leftmost number, so the order the eye reads matches the order the
    # rows are in. Sorting by the raw count instead put the table in %Team order — the two are
    # proportional — which left Act/90 jumping around down the column for no visible reason.
    #
    # The cap therefore selects on rate too. Measured on Nashville that barely moves the
    # membership (one 1-action cameo in, one 2-action ever-present out) because a rate this low
    # needs real minutes behind it to place, and the thin-minutes rows it does promote are the
    # ones the greying below is there to mark.
    ranked = sorted(((player, actions, player_minutes.get(player, 0))
                     for player, actions in counts.items()),
                    key=lambda row: row[1] / row[2] * 90 if row[2] else 0, reverse=True)

    values, grey = [], []
    for player, actions, minutes in ranked[:CORNER_LIST_MAX]:
        per_90 = actions / minutes * 90 if minutes else 0
        values.append((abbreviate_player_name(player), f'{per_90:.2f}', actions))
        grey.append(not qualified(player))

    header, rows = format_key_table(
        [('Player', 17, '<'), ('Act/90', 7, '>'), ('Act', 5, '>')], values)
    return header, rows, grey

# Mirrors generate_corner_map's bbox on the right half of the page, at the same y and the same
# size — the pair reads as a pair. Its own key hangs in the SAME central column as that chart's,
# below them, which is why it takes key_column_bottom rather than placing itself: there is no room
# for a text list to the right of a chart whose colourbar label already reaches x~0.95.
def generate_def_corner_map(leagues, team, matches, date, team_colors, page,
                            bbox=[0.60, 0.525, 0.208, 0.26], key_column=None,
                            expected_lineup=None, expected_subs=None, shared_vmax=None):
    '''Where this team CONCEDES corners, and who deals with them — the defensive mirror of
    generate_corner_map.

    Same encodings as its sibling minus the taker: these corners are struck by whoever the team
    happened to face, so a per-taker key would list strangers from a dozen clubs. Flat
    team_colors[1], the same treatment the two other defensive mirrors use. Arc shape and
    hollow-when-cleared are kept and are decoded by the Delivery key beside the sibling chart.

    Returns (page, notes_bbox).'''

    # Corners conceded live in the OPPONENTS' own event files — unlike shots, which Shot_Map.csv
    # carries for both sides. Same cross-reference generate_def_key_pass_map relies on.
    opponent_matches = load_actual_opponent_events(matches, leagues, date, team)
    corners = extract_corners(opponent_matches)
    if not corners.empty:
        valid_end = np.isfinite(corners['Pass End X Coord']) & np.isfinite(corners['Pass End Y Coord'])
        corners = corners[valid_end].copy()

    if corners.empty:
        print('No conceded corner data available — skipping conceded corner map.')
        return page, None

    # VALUE FIRST, ROTATE SECOND — both of these joins measure distances against data still in
    # the delivering team's own frame, and both would silently mismatch if run after the rotation.
    xg_created = corner_xg_created(corners, opponent_matches, matches, team, conceded=True)
    defender_counts = corner_defensive_actions(corners, opponent_matches, matches)
    xg_available = has_xg(extract_shots(matches, team, conceded=True, quiet=True))

    rotate_into_opponent_frame(corners, [('Player X Coord', 'Player Y Coord'),
                                         ('Pass End X Coord', 'Pass End Y Coord')])

    # Arc has to be RE-DERIVED from the rotated coordinates. extract_corners set it from the
    # taker's own frame, where the arc at y~0 is the bottom one; after rotating, that same
    # delivery is drawn at the TOP of this chart. Leaving it would make the shape a reader sees
    # here disagree with the key that decodes it on the chart beside this one. Threshold 50 rather
    # than 1, since the rotated arcs sit at y~99.5 and y~0.5.
    corners['Arc'] = corners['Player Y Coord'] < 50

    ax = page.add_axes(bbox)
    # pitch_color='none' so the pitch's own background matches the page instead of
    # painting a separate fill behind it — same as every other chart in this file.
    pitch = Pitch(pitch_type='opta', pitch_color='none', line_color='#888888', linewidth=1)
    pitch.draw(ax=ax)
    crop_corner_map(ax, mirrored=True)

    mesh = None
    if xg_available:
        xg_bins = pitch.bin_statistic(corners['Pass End X Coord'], corners['Pass End Y Coord'],
                                      values=xg_created, statistic='sum', bins=CORNER_HEATMAP_BINS)

        cmap = LinearSegmentedColormap.from_list('cmap', ['#ffffff', team_colors[0]])
        # From the offensive sibling, which is the only chart of the pair holding both directions'
        # data. Falls back to this chart's own maximum only when that chart was skipped.
        max_val = shared_vmax if shared_vmax else np.nanmax(xg_bins['statistic'])
        if not max_val or not np.isfinite(max_val):
            max_val = 1

        mesh = pitch.heatmap(xg_bins, ax=ax, cmap=cmap, vmin=0, vmax=max_val, zorder=1,
                             alpha=SHOT_HEATMAP_ALPHA)

        existing_artists = set(ax.get_children())
        pitch.draw(ax=ax)
        crop_corner_map(ax, mirrored=True)
        for artist in ax.get_children():
            if artist not in existing_artists:
                artist.set_zorder(mesh.get_zorder() + 1)

    plot_corner_marks(ax, corners, team_colors[1])

    pitch_pos = ax.get_position()
    if mesh is not None:
        cbar_ax = page.add_axes([pitch_pos.x1 + 0.02, pitch_pos.y0, 0.015, pitch_pos.height])
        cbar = page.colorbar(mesh, cax=cbar_ax)
        cbar.set_label('ΣxG Conceded / Zone (first contact)', labelpad=6, fontsize=8)

    ax.set_title('Corners Conceded' if xg_available
                 else f'Corners Conceded ({SHOT_XG_FALLBACK_NOTE})', fontsize=10, pad=12)
    # Arrow points LEFT because the opposition attacks toward x=0 in this rotated frame — the
    # visible cue that this is the defensive mirror, matching the pattern on page three.
    ax.set_xlabel('←    Opposition Attack    ←', labelpad=8, fontsize=8)

    # The defender list is drawn on the SIBLING chart's axes, in the shared central column: a
    # legend is positioned in its own axes' fractions, and placing it from here would need a
    # large negative anchor that breaks the moment either bbox moves.
    defender_header, defender_rows, defender_grey = corner_defender_rows(
        defender_counts, corners, matches, expected_lineup, expected_subs)
    if defender_rows and key_column is not None:
        anchor_ax, column_bottom = key_column
        anchor_pos = anchor_ax.get_position()
        top = (column_bottom - anchor_pos.y0) / anchor_pos.height - SHOT_LEGEND_GAP
        table_legend(anchor_ax, defender_header, defender_rows, 'Corners Defended',
                     CORNER_LEGEND_ANCHOR_X, top, grey=defender_grey)

    notes_x0 = 0.75 - NOTES_WIDTH / 2
    notes_bbox = draw_notes_box(page, [notes_x0, CORNER_NOTES_Y0, NOTES_WIDTH,
                                       bbox[1] - 0.03 - CORNER_NOTES_Y0])

    return page, notes_bbox

#%%
def main():


    return

if __name__ == "__main__":
    main()