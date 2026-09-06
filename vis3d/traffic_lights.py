"""
Traffic-light helpers shared by the vis3d pipeline: recognising a
GroundingDINO detection as a traffic light, reading its lit state (red /
yellow / green) off the frame pixels, and the fixed cuboid dimensions the
renderer draws it with.

The state has to come from the image itself: unlike the navsim/nuPlan path
(navsim/visualization/raster.py), which reads a logged per-lane-connector
traffic-light status out of the map database, raw video frames carry no such
annotation -- the only evidence of the state is which lamp of the housing is
lit. So this looks for the lit lamp in the mask's own pixels, which is what a
human reads the light by too.

Finding the lamp is the whole problem, and counting coloured pixels does not
do it. This used to take the mask's brightest, most saturated pixels as lamp
candidates and let them vote by hue, on the stated assumption that "the
housing itself is dark and its pixels carry no colour evidence". That is false
wherever the housing is painted: New York hangs its lights in bright yellow
bodies, and in sun those are both brighter and more saturated than the lamp
lens, so the vote counts the paintwork. On the ambulance clip it returned
"yellow" for 136 of 142 detections and green for none, through an intersection
the ego drives straight through on green.

What separates a lamp from a body is shape, not intensity: a lit lens is one
small compact blob, while the housing is spread over the whole detection. So
each colour is scored by the largest connected blob of its own hue -- how far
it reaches along the head's long axis and how solidly it fills its own box --
and the body is disqualified outright by hue, since whatever colour covers
most of the mask is by definition not a lamp. What survives is conservative:
on that clip, 50 greens, no reds or yellows, and "unknown" for the other 92
(distant, blurred or seen from the side). Being unlabelled costs little here,
because smooth_boxes.py votes each light's state over its whole track and a
handful of decided frames carries the rest.

Dimensions are fixed rather than fitted (RAP models traffic lights as upright
cuboids with fixed dimensions, and the same constants live in
visualization/renderer.py's ScenarioRenderer for the navsim path): a traffic
light is a small, thin, self-similar object, so a mask-fitted box is dominated
by monocular-depth noise -- the mask is a few dozen pixels at 30-80 m, where
the depth band those pixels span is far wider than the housing itself.
"""
import cv2
import numpy as np

# (length, width, height) in metres, matching renderer.py's navsim-path cuboid.
TRAFFIC_LIGHT_DIMS = (0.5, 0.5, 1.0)

STATES = ("red", "yellow", "green", "unknown")

# Hue bands in OpenCV's 0-179 convention. Red wraps around the end of the
# circle, hence two bands. The bands deliberately leave gaps (95-170 = blue /
# violet): a light against the sky leaks sky through its mask, and leaving that
# hue out of every band is what stops it standing for a lamp -- there is no
# band for it to be the largest blob of.
_HUE_BANDS = {
    "red": ((0, 12), (168, 180)),
    "yellow": ((13, 36),),
    "green": ((37, 95),),
}

# A lamp lens is a strongly coloured, reasonably bright thing. These floors are
# absolute rather than relative to the mask's own peak, because the peak here is
# usually the sunlit housing: scaling to it would move the bar with the very
# thing being excluded.
MIN_LAMP_SATURATION = 70
MIN_LAMP_VALUE = 70

# The smallest blob that is allowed to decide a state. A lens is a handful of
# pixels at 40 m and this is what stops a stray speck of leaked background from
# naming the light; below it the answer is "unknown".
MIN_LAMP_PIXELS = 6

# What makes a blob lamp-shaped. A three-lens head is drawn as its longest
# dimension, and one lens covers about a third of it, so a blob reaching further
# than MAX_LAMP_SPAN along that axis is bodywork or a leak and not a lens. Fill
# is the blob's area over its own bounding box: a lens is a disc (~0.79) while a
# body outline threading round a visor is far more ragged.
MAX_LAMP_SPAN = 0.45
MIN_LAMP_FILL = 0.30

# Whatever hue covers most of the mask is the housing, not a lamp -- a lens is
# always a small part of the head it sits in. Any band the housing hue falls in
# (or within HOUSING_HUE_PAD of, since the paint's hue wanders with exposure and
# shadow) is therefore unusable, and the state is read from what is left.
#
# HOUSING_SHARE keeps this off the case it would get wrong. A dark housing has
# no hue worth the name: its modal hue is noise off near-black pixels, spread
# thin, so its share never reaches the threshold and no band is disqualified --
# which is exactly the old assumption, still honoured where it holds. The
# disqualification only fires when the mask really is dominated by one colour.
HOUSING_SHARE = 0.15
HOUSING_HUE_PAD = 14

# Width of the box smoothing the hue histogram before its mode is taken, in hue
# units. Painted bodywork is one colour spread over a few degrees by shading and
# JPEG chroma, so the raw argmax lands on whichever single degree happened to win.
_HOUSING_HUE_SMOOTHING = 9


def is_traffic_light(category: str) -> bool:
    """True for the detection categories that name a traffic light.

    Matched loosely because the category string is whatever phrase
    GroundingDINO returned for the box, which is not guaranteed to be the
    prompt term verbatim -- it can come back split or partial (e.g. "traffic"
    alone) depending on how the prompt tokenised.
    """
    category = category.lower()
    return "traffic light" in category or category.strip() in ("traffic", "trafficlight")


def _housing_hue(hue: np.ndarray, mask: np.ndarray):
    """(modal hue of the mask, the share of the mask it holds).

    Circular: the histogram is tiled three times before smoothing so a mode
    sitting on the red wrap-around is not cut in half by the ends of the array.
    """
    histogram = np.bincount(hue[mask], minlength=180).astype(np.float64)
    kernel = np.ones(_HOUSING_HUE_SMOOTHING) / _HOUSING_HUE_SMOOTHING
    smoothed = np.convolve(np.tile(histogram, 3), kernel, mode="same")[180:360]
    mode = int(np.argmax(smoothed))
    # Undo the kernel's averaging to read the share back as a pixel count.
    share = smoothed[mode] * _HOUSING_HUE_SMOOTHING / max(int(mask.sum()), 1)
    return mode, float(share)


def _band_is_housing(bands, housing_hue: int, housing_share: float) -> bool:
    """True if `bands` is the colour of the bodywork rather than of a lamp."""
    if housing_share < HOUSING_SHARE:
        return False
    if any(low <= housing_hue <= high for low, high in bands):
        return True
    return min(min(abs(housing_hue - low), abs(housing_hue - high))
               for low, high in bands) <= HOUSING_HUE_PAD


def _lamp_blob(candidates: np.ndarray, long_axis_px: int, vertical: bool):
    """(area, span, fill) of the largest connected blob in `candidates`, or None.

    `span` is how far the blob reaches along the head's long axis as a fraction
    of it, and `fill` how much of its own bounding box it occupies -- the two
    numbers that say "one lens" rather than "the whole body".
    """
    count, _, stats, _ = cv2.connectedComponentsWithStats(
        candidates.astype(np.uint8), connectivity=8)
    if count < 2:                      # label 0 is the background
        return None
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    _, _, width, height, area = stats[largest]
    if area < MIN_LAMP_PIXELS:
        return None
    span = (height if vertical else width) / max(long_axis_px, 1)
    return int(area), float(span), area / float(max(width * height, 1))


def classify_state(image_bgr: np.ndarray, mask: np.ndarray) -> str:
    """Reads the lit state of the traffic light covered by `mask`.

    :param image_bgr: the full frame, BGR uint8 (as cv2.imread returns it)
    :param mask: boolean mask of the traffic light, same HxW as image_bgr
    :return: one of STATES; "unknown" when no lamp is convincingly lit, which
        the renderer draws in white rather than guessing a colour.
    """
    rows, cols = np.nonzero(mask)
    if rows.size == 0:
        return "unknown"
    height = int(rows.max() - rows.min()) + 1
    width = int(cols.max() - cols.min()) + 1
    # Which way the head is mounted, so "along the head" means the same thing
    # for a vertical three-lens body and a horizontal one.
    vertical = height >= width
    long_axis_px = height if vertical else width

    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0].astype(np.int32)
    lamp_like = mask & (hsv[:, :, 1] >= MIN_LAMP_SATURATION) \
                     & (hsv[:, :, 2] >= MIN_LAMP_VALUE)
    housing_hue, housing_share = _housing_hue(hue, mask)

    best_score, best_state = 0.0, "unknown"
    for state, bands in _HUE_BANDS.items():
        if _band_is_housing(bands, housing_hue, housing_share):
            continue
        in_band = lamp_like & np.any([(hue >= low) & (hue <= high)
                                      for low, high in bands], axis=0)
        if np.count_nonzero(in_band) < MIN_LAMP_PIXELS:
            continue
        blob = _lamp_blob(in_band, long_axis_px, vertical)
        if blob is None:
            continue
        area, span, fill = blob
        if span > MAX_LAMP_SPAN or fill < MIN_LAMP_FILL:
            continue
        # Bigger, rounder and tighter all argue for a lens; between two bands
        # that both pass, this is what ranks the greener half of an overexposed
        # lamp against the warm fringe its own bloom leaves on the visor.
        score = area * fill / max(span, 1e-3)
        if score > best_score:
            best_score, best_state = score, state
    return best_state
