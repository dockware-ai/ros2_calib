#!/usr/bin/env python3
"""Render a printable AprilGrid board as a PNG.

Geometry follows the AprilGrid/Kalibr convention, which is also what the intrinsics
capture tool assumes:

    tag size    the outer edge of the BLACK square = 8 cells for 36h11
                (6x6 data bits + a 1-cell black border on every side)
    spacing     the white gap between neighbouring black squares, as a fraction of
                tag size. 0.25 -> 2 cells, which doubles as the quiet zone the
                detector needs around each tag
    pitch       tag + gap = 10 cells = 1.25 x tag size

Tags are drawn in the reference AprilTag orientation (the one apriltag-imgs and every
published tag36h11 image use) and ids run row-major from the BOTTOM-left, as on a Kalibr
board: id = (rows - 1 - row) * cols + col + offset, with row counted down the sheet. That
pairing is what calibrate.py's tag_world_corners() models -- the detector's corner order
(bottom-left, bottom-right, top-right, top-left of an upright tag) lands on its object
points only with both. The inner-4x4 keep_tag_ids [7..28] still sit in the middle of a
6x6, the set is symmetric.

OpenCV's DICT_APRILTAG_36h11 decodes to the same ids but draws every marker rotated 180
degrees from the reference. The detector does not care about the rotation, so such a
board reads back with the right ids -- and with every corner on the opposite side of its
tag, a full tag-size error in each correspondence. render() turns the OpenCV patch back
upright, and verify() checks corner order, not just positions.

Cells are whole pixels, so nothing is resampled and every edge lands exactly on a pixel
boundary. The PNG carries its DPI, so printing at 100% (no "scale to fit") gives the
physical size in the caption -- check it against the printed scale bar before trusting a
calibration made from it.

    python3 make_aprilgrid.py                       # 6x6, tag 112.1mm, matches the
                                                    # existing Tangram board
    python3 make_aprilgrid.py --tag-mm 40 --dpi 600 # small board on one sheet
"""

import argparse

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

FAMILY = "tag36h11"
DATA_BITS = 6  # 36h11 -> 6x6 data grid
BORDER_CELLS = 1  # black ring around the data, part of "tag size"
TAG_CELLS = DATA_BITS + 2 * BORDER_CELLS


def render(
    rows,
    cols,
    spacing,
    id_offset,
    tag_mm,
    dpi,
    margin_mm,
    caption_mm,
    checker,
    ring=True,
    id_cols=None,
):
    """Draw the board, returning (image, geometry dict). White background, black tags."""
    cell_px = max(1, round(tag_mm / TAG_CELLS / 25.4 * dpi))
    tag_px = TAG_CELLS * cell_px
    gap_px = round(spacing * tag_px)
    pitch_px = tag_px + gap_px
    # With the checker ring the outermost ink sits one gap beyond the tags, so the
    # margin floor doubles to keep a full gap of quiet zone outside them.
    ring = checker and ring
    ring_px = gap_px if ring else 0
    margin_px = max(round(margin_mm / 25.4 * dpi), ring_px + gap_px)
    # caption_mm < 0 asks for the caption inside the margin (square sheet): make sure the
    # white band outside the ink is deep enough to hold a line of text or the scale bar.
    if caption_mm < 0:
        margin_px = max(margin_px, ring_px + round(-caption_mm / 25.4 * dpi))
        caption_mm = 0.0

    board_w = cols * tag_px + (cols - 1) * gap_px
    board_h = rows * tag_px + (rows - 1) * gap_px
    # Extra room under the board for the caption and the scale bar.
    caption_px = round(caption_mm / 25.4 * dpi)
    image = np.full((board_h + 2 * margin_px + caption_px, board_w + 2 * margin_px), 255, np.uint8)

    tags = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    placed = {}
    for row in range(rows):
        for col in range(cols):
            # id_cols > cols renders a window of a wider board with that board's ids.
            tag_id = (rows - 1 - row) * (id_cols or cols) + col + id_offset
            # OpenCV's markers are the reference tags rotated 180 degrees; undo that.
            patch = np.rot90(cv2.aruco.generateImageMarker(tags, tag_id, tag_px, BORDER_CELLS), 2)
            x, y = margin_px + col * pitch_px, margin_px + row * pitch_px
            image[y : y + tag_px, x : x + tag_px] = patch
            placed[tag_id] = (col, row)

    # Gap-sized black squares on the junction lattice, (cols+1) x (rows+1) of them: one
    # in every gap crossing and a ring one pitch outside the tags. They are what makes
    # this a Tangram-style AprilGrid rather than a bare Kalibr grid -- the tags give
    # identity, these give sharp chessboard corners between them. Without the ring only
    # the interior crossings are inked, so the outer edge of the ink is the tags' own
    # edge -- a clean rectangle for a LiDAR to find the board boundary against.
    if checker:
        lo = 0 if ring else 1
        for i in range(lo, cols + 1 - lo):
            for j in range(lo, rows + 1 - lo):
                x = margin_px + i * pitch_px - gap_px
                y = margin_px + j * pitch_px - gap_px
                image[y : y + gap_px, x : x + gap_px] = 0

    # Report what is actually printed, not what was asked for: snapping cells to whole
    # pixels moves the tag size by up to half a cell, and that error is a scale error in
    # every calibration made from the board.
    mm = 25.4 / dpi
    return image, {
        "cell_px": cell_px,
        "tag_px": tag_px,
        "gap_px": gap_px,
        "pitch_px": pitch_px,
        "ring_px": ring_px,
        "margin_px": margin_px,
        "caption_px": caption_px,
        "x0": margin_px,
        "y0": margin_px,  # top-left tag's outer corner, in pixels
        "placed": placed,
        "spacing": gap_px / tag_px,
        "tag_mm": tag_px * mm,
        "gap_mm": gap_px * mm,
        "pitch_mm": pitch_px * mm,
        "board_w_mm": board_w / dpi * 25.4,
        "board_h_mm": board_h / dpi * 25.4,
        "ink_w_mm": (board_w + 2 * ring_px) / dpi * 25.4,
        "ink_h_mm": (board_h + 2 * ring_px) / dpi * 25.4,
        "sheet_w_mm": image.shape[1] / dpi * 25.4,
        "sheet_h_mm": image.shape[0] / dpi * 25.4,
    }


def annotate(image, geom, rows, cols, id_offset, dpi, mode, label=""):
    """Caption the sheet with everything a calibration needs, plus a scale bar.

    A board whose printed tag size is unknown cannot be calibrated from, and a printer
    that quietly scaled to fit makes the number in the filename a lie -- so the numbers
    travel on the sheet and the bar is there to measure.

    mode 'full' prints both under the board, 'bar' only the scale bar, 'margin' only the
    family/grid/ids line (no sizes -- the printed board is measured, not trusted), in
    bold inside the white border above the board so the sheet stays
    square, 'none' leaves the sheet bare -- bare means nothing on the paper can tell you
    what it was scaled to, so the tag size has to be measured by hand before the board
    is worth calibrating from.
    """
    if mode == "none":
        return image
    pil = Image.fromarray(image)
    draw = ImageDraw.Draw(pil)
    # Margin mode has a whole border to itself and is read on a mounted board from a
    # step back, so it goes larger and bold.
    size = max(12, round((7.0 if mode == "margin" else 4.5) / 25.4 * dpi))
    face = "DejaVuSansMono-Bold.ttf" if mode == "margin" else "DejaVuSansMono.ttf"
    try:
        font = ImageFont.truetype(face, size)
    except OSError:
        font = ImageFont.load_default(size)

    left = geom["margin_px"]
    top = image.shape[0] - geom["caption_px"] - round(2 / 25.4 * dpi)
    # Margin mode: centre each line in the white band between the sheet edge and the ink.
    band = geom["margin_px"] - geom["ring_px"]
    ids = id_text(geom)
    if mode in ("full", "margin"):
        sizes = (
            []
            if mode == "margin"
            else [
                f"tag {geom['tag_mm']:.2f}mm",
                f"gap {geom['gap_mm']:.2f}mm",
                f"pitch {geom['pitch_mm']:.2f}mm",
            ]
        )
        text = "  ".join(
            ([label] if label else [])
            + [f"{FAMILY} {cols}x{rows}"]
            + sizes
            + [f"ids {ids} row-major from bottom-left"]
            + ([] if mode == "margin" else [f"{dpi}dpi"])
        )
        if mode == "margin":
            draw.text((left, band // 2), text, fill=0, font=font, anchor="lm")
            return np.asarray(pil)
        draw.text((left, top), text, fill=0, font=font)

    # 100mm reference: print at 100% and this measures 100mm. If it does not, the sheet
    # was scaled and every dimension above is wrong by the same factor.
    bar_y = (top + round(9 / 25.4 * dpi)) if mode == "full" else top
    # A 100mm bar does not fit a small sheet; fall back to 50 or 20 so the reference is
    # still a round number someone can lay a ruler against.
    room_mm = (image.shape[1] - 2 * left) / dpi * 25.4
    bar_mm = next((b for b in (100, 50, 20) if b + 30 <= room_mm), 20)
    bar_px = round(bar_mm / 25.4 * dpi)
    tick = round(2.5 / 25.4 * dpi)
    draw.line([(left, bar_y), (left + bar_px, bar_y)], fill=0, width=max(2, dpi // 150))
    for x in (left, left + bar_px):
        draw.line([(x, bar_y - tick), (x, bar_y + tick)], fill=0, width=max(2, dpi // 150))
    bar_text = (
        f"{bar_mm}mm -- print at 100%, do not scale to fit" if mode == "full" else f"{bar_mm}mm"
    )
    draw.text((left + bar_px + tick * 2, bar_y), bar_text, fill=0, font=font, anchor="lm")
    return np.asarray(pil)


def id_text(geom):
    """'0..35' for a contiguous id block, '0 1 8 9 16 17' for a window of a wider board."""
    ids = sorted(geom["placed"])
    if ids[-1] - ids[0] + 1 == len(ids):
        return f"{ids[0]}..{ids[-1]}"
    return " ".join(map(str, ids))


PAGES_MM = {"letter": (215.9, 279.4), "a4": (210.0, 297.0)}


def test_page(image, geom, page, dpi, label, board_desc):
    """Centre a bare tag window on a printer page, with expected sizes and a scale bar.

    For checking a desk printer against a big board before ordering it: the tags are the
    big board's exact size and ids, and the caption says what a caliper should read.
    """
    page_w, page_h = (round(v / 25.4 * dpi) for v in PAGES_MM[page])
    mm = dpi / 25.4
    top = round(12 * mm)
    if image.shape[1] > page_w - round(12 * mm) or image.shape[0] + top + round(40 * mm) > page_h:
        raise SystemExit(
            f"{image.shape[1] / mm:.0f} x {image.shape[0] / mm:.0f} mm of tags "
            f"does not fit a {page} page with room for the caption"
        )
    out = np.full((page_h, page_w), 255, np.uint8)
    left = (page_w - image.shape[1]) // 2
    out[top : top + image.shape[0], left : left + image.shape[1]] = image
    geom = dict(
        geom,
        x0=geom["x0"] + left,
        y0=geom["y0"] + top,
        sheet_w_mm=PAGES_MM[page][0],
        sheet_h_mm=PAGES_MM[page][1],
    )

    pil = Image.fromarray(out)
    draw = ImageDraw.Draw(pil)
    size = round(3.5 * mm)
    try:
        font = ImageFont.truetype("DejaVuSansMono.ttf", size)
        bold = ImageFont.truetype("DejaVuSansMono-Bold.ttf", size)
    except OSError:
        font = bold = ImageFont.load_default(size)
    x = geom["x0"]
    y = top + image.shape[0] + round(4 * mm)
    lines = [
        (bold, f"{label + '  ' if label else ''}TEST PAGE -- {board_desc}"),
        (font, f"{FAMILY}  ids {id_text(geom)}  row-major from bottom-left"),
        (
            bold,
            f"expect: tag {geom['tag_mm']:.2f}mm  gap {geom['gap_mm']:.2f}mm  "
            f"pitch {geom['pitch_mm']:.2f}mm",
        ),
        (font, "print at 100% / actual size, never fit to page"),
    ]
    for f, text in lines:
        draw.text((x, y), text, fill=0, font=f)
        y += round(5.5 * mm)
    y += round(3 * mm)
    bar = round(100 * mm)
    tick = round(2.5 * mm)
    w = max(2, dpi // 150)
    draw.line([(x, y), (x + bar, y)], fill=0, width=w)
    for bx in (x, x + bar):
        draw.line([(bx, y - tick), (bx, y + tick)], fill=0, width=w)
    draw.text((x + bar + 2 * tick, y), "100mm", fill=0, font=font, anchor="lm")
    return np.asarray(pil), geom


def verify(image, geom, rows, cols):
    """Detect the rendered board and check it reads back as the board we meant to draw.

    Rendering a target is the one step nobody checks and everybody pays for later: a
    dictionary whose ids do not match the reference family, or a layout mirrored against
    the calibrator's, both look perfectly fine on paper.
    """
    from pupil_apriltags import Detector

    # Full resolution, no quad_decimate: the detector's decimation is a nearest-neighbour
    # shortcut that fuses a tag's black border with the checker square touching its
    # corner, dropping tags a camera resolves perfectly well. Under a second even on a
    # 100MP sheet, so there is nothing to buy by approximating here.
    detector = Detector(families=FAMILY, nthreads=4, quad_decimate=1.0)
    found = detector.detect(image)
    # pupil-apriltags' destructor double-frees the family and segfaults at exit; same
    # defusal as calibrate.py's release_detectors().
    try:
        detector.tag_families.clear()
    except AttributeError:
        pass
    report = [f"detected {len(found)}/{rows * cols} tags"]
    ok = len(found) == rows * cols

    by_id = {d.tag_id: np.asarray(d.corners, float) for d in found}
    missing = sorted(set(geom["placed"]) - set(by_id))
    extra = sorted(set(by_id) - set(geom["placed"]))
    if missing or extra:
        ok = False
        report.append(f"ids missing {missing} unexpected {extra}")

    # Every corner must land where its id and index say. Checking tag centres is not
    # enough: a tag drawn rotated or mirrored keeps its id and centre but moves every
    # corner, which is exactly the mistake that ruins a calibration silently.
    # Detector order on an upright tag: bottom-left, bottom-right, top-right, top-left.
    misplaced = []
    t = geom["tag_px"]
    for tag_id, quad in by_id.items():
        if tag_id not in geom["placed"]:
            continue
        col, row = geom["placed"][tag_id]
        x0 = geom["x0"] + col * geom["pitch_px"]
        y0 = geom["y0"] + row * geom["pitch_px"]
        want = np.array([(x0, y0 + t), (x0 + t, y0 + t), (x0 + t, y0), (x0, y0)], float)
        if np.abs(quad - want).max() > geom["cell_px"]:
            misplaced.append(tag_id)
    if misplaced:
        ok = False
        report.append(f"tags misplaced or misoriented: {misplaced}")
    else:
        report.append("every corner sits where its id and corner index say")

    # Measured tag/pitch ratio, against the geometry the capture tool is configured for.
    first = min(geom["placed"])
    if first in by_id and first + 1 in by_id:
        side = np.linalg.norm(by_id[first][0] - by_id[first][1])
        step = np.linalg.norm(by_id[first + 1].mean(axis=0) - by_id[first].mean(axis=0))
        measured = side / 2.0 / step
        expected = 0.5 / (1.0 + geom["spacing"])
        report.append(f"measured tag_half_pitch {measured:.3f} (expected {expected:.3f})")
        ok = ok and abs(measured - expected) < 0.02
    return ok, report


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--cols", type=int, default=6, help="marker grid width")
    parser.add_argument("--rows", type=int, default=6, help="marker grid height")
    parser.add_argument(
        "--spacing", type=float, default=0.25, help="gap between tags as a fraction of tag size"
    )
    parser.add_argument("--id-offset", type=int, default=0, help="id of the bottom-left tag")
    parser.add_argument(
        "--tag-mm",
        type=float,
        default=112.1,
        help="printed size of one tag (black square edge). The default "
        "matches the Tangram 6x6 board already in the capture config",
    )
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--margin-mm", type=float, default=15.0)
    parser.add_argument(
        "--no-checker",
        dest="checker",
        action="store_false",
        help="omit the black squares between and around the tags, leaving a bare Kalibr-style grid",
    )
    parser.add_argument(
        "--no-ring",
        dest="ring",
        action="store_false",
        help="keep the black squares between tags but drop the ring "
        "outside them, so the ink ends at the tags (LiDAR boards)",
    )
    parser.add_argument(
        "--caption",
        choices=("full", "bar", "margin", "none"),
        default="full",
        help="'full' geometry line + scale bar under the board, 'bar' "
        "just the scale bar, 'margin' a bold geometry line inside "
        "the white border so the sheet stays square, 'none' a bare board -- with "
        "'none' nothing on the paper proves the printer did not "
        "rescale it",
    )
    parser.add_argument(
        "--label", default="", help='name printed at the start of the caption, e.g. "board B"'
    )
    parser.add_argument(
        "--id-cols",
        type=int,
        default=None,
        help="column count of the board the ids are numbered on; set it "
        "wider than --cols to render a window of that board",
    )
    parser.add_argument(
        "--page",
        choices=sorted(PAGES_MM),
        default=None,
        help="lay the tags out as a desk-printer test page (caption with "
        "expected sizes + scale bar); also writes a PDF next to --out",
    )
    parser.add_argument(
        "--board-desc", default="", help="test page only: what board these tags stand in for"
    )
    parser.add_argument("--out", default="/tmp/aprilgrid_36h11_6x6.png")
    args = parser.parse_args()
    if args.page:
        args.caption = "none"

    # Negative: no strip under the board, the margin is deepened to hold this much text.
    caption_mm = {"full": 28.0, "bar": 12.0, "margin": -12.0, "none": 0.0}[args.caption]
    image, geom = render(
        args.rows,
        args.cols,
        args.spacing,
        args.id_offset,
        args.tag_mm,
        args.dpi,
        args.margin_mm,
        caption_mm,
        args.checker,
        args.ring,
        args.id_cols,
    )
    image = annotate(
        image, geom, args.rows, args.cols, args.id_offset, args.dpi, args.caption, args.label
    )
    if args.page:
        image, geom = test_page(image, geom, args.page, args.dpi, args.label, args.board_desc)
    # Verify the finished sheet, caption included: text must not read back as a tag.
    ok, report = verify(image, geom, args.rows, args.cols)

    print(f"{args.out}")
    print(f"  {FAMILY}  {args.cols}x{args.rows}  ids {id_text(geom)}")
    print(
        f"  tag {geom['tag_mm']:.3f}mm   gap {geom['gap_mm']:.3f}mm   "
        f"pitch {geom['pitch_mm']:.3f}mm   spacing {geom['spacing']:g}"
    )
    if abs(geom["tag_mm"] - args.tag_mm) > 0.005:
        print(
            f"  note: asked for {args.tag_mm:g}mm tags, whole-pixel cells at {args.dpi}dpi "
            f"give {geom['tag_mm']:.3f}mm -- calibrate with the printed value"
        )
    print(
        f"  board {geom['board_w_mm']:.1f} x {geom['board_h_mm']:.1f} mm (tags)"
        + (
            f"   {geom['ink_w_mm']:.1f} x {geom['ink_h_mm']:.1f} mm incl. checker ring"
            if geom["ring_px"]
            else "   ink ends at the tags (no ring)"
        )
    )
    print(
        f"  sheet {geom['sheet_w_mm']:.1f} x {geom['sheet_h_mm']:.1f} mm "
        f"at {args.dpi}dpi -> {image.shape[1]}x{image.shape[0]} px"
    )
    print(f"  cell {geom['cell_px']}px (whole pixels, no resampling)")
    for line in report:
        print(f"  verify: {line}")
    print(
        f"  capture config for this board: grid_cols {args.id_cols or args.cols}, "
        f"tag_half_pitch {0.5 / (1.0 + geom['spacing']):g}"
    )
    if not ok:
        raise SystemExit("verification FAILED -- not saved, do not print this")
    Image.fromarray(image).save(args.out, dpi=(args.dpi, args.dpi))
    if args.page:
        # A PDF carries its page size, so "actual size" printing cannot rescale it the
        # way image viewers do with a PNG.
        pdf = args.out.rsplit(".", 1)[0] + ".pdf"
        # Bilevel: lossless in the PDF (and this Pillow has no JPEG encoder for greyscale).
        Image.fromarray(image).point(lambda v: 255 if v >= 128 else 0).convert("1").save(
            pdf, resolution=float(args.dpi)
        )
        print(f"  pdf: {pdf}")


if __name__ == "__main__":
    main()
