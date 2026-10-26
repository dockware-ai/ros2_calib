#!/usr/bin/env python3
"""Camera intrinsic calibration from multi-board AprilGrid (tag36h11) images.

Drop-in replacement for calibrate.py for the four-sheet 8x8 board set that
make_aprilgrid.py renders into aprilgrid_23in_8x8/ (board A ids 0-63, B 64-127,
C 128-191, D 192-255, each row-major from its bottom-left). The sheets may lie anywhere,
at any pose relative to each other, overlapping or not. Same command line and the same
camera_info YAMLs in the device-config layout:

    devices/<serial>/ip_camera_processing_cpp/
        context_camera_info.yaml        zoom_camera_info.yaml
        debug_context_camera_info.yaml  debug_zoom_camera_info.yaml

Usage:
    python aprilgrid_detect.py --serial vss_00000041 --context <image_dir>
    python aprilgrid_detect.py --serial vss_00000041 --zoom <image_dir>
    python aprilgrid_detect.py --serial vss_00000041 --context <dir> --zoom <dir>

Detection, per image:

    pass 1   pupil-apriltags over the whole frame at decimation 1 and 2, on a CLAHE copy
             (glare) and on 2x/4x downscales (large defocused tags close to the camera).
             Only exact (hamming 0) decodes are kept
    pass 2   per board, every cell next to a found tag is predicted with a homography of
             its found neighbours, warped to a fronto-parallel patch (flattening grazing
             views, enlarging small tags) and decoded there. Only the id that belongs in
             that cell is accepted, so a few bit errors are tolerated. Repeats until the
             board stops growing. Cells covered by another sheet are skipped
    refine   every tag is rectified through its own corners and its four outer edges are
             fitted as sub-pixel lines, which are intersected for the corners

Board geometry is worked in undistorted pixels -- from --initial-guess, else a pinhole
guess from the EXIF 35 mm focal length -- so a flat sheet is exactly a homography. Tags
that are cut by the image border are not used.

Each sheet seen in an image is its own calibration view (its own pose), so sheets need
not be coplanar. Calibration and outlier rejection are calibrate.py's.
"""

import argparse
import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
import yaml
from PIL import Image
from pupil_apriltags import Detector

from calibrate import (
    CAMERA_SUBDIR,
    CAMERAS,
    DEVICES_DIRNAME,
    _project_fisheye,
    _project_zoom,
    _scale_K,
    calibrate_fisheye,
    calibrate_robust,
    calibrate_zoom,
    camera_info,
    confirm_overwrite,
    load_prior_calibration,
    normalize_serial,
    output_paths,
    print_view_errors,
    write_yaml,
)

# ── Board layout (aprilgrid_23in_8x8: four 8x8 tag36h11 sheets) ─────────────
TAG_FAMILY = "tag36h11"
GRID_COLS = 8
GRID_ROWS = 8
NUM_BOARDS = 4
TAGS_PER_BOARD = GRID_COLS * GRID_ROWS  # board b holds ids b*64 .. b*64+63
PITCH = 1.25  # tag + gap, in tag edges (spacing 0.25)
# Black-square edge as rendered (664 px at 300 dpi). Intrinsics do not depend on it;
# only the scale of the solved board poses does.
TAG_SIZE = 0.0562

BOARD_NAMES = "ABCD"
BOARD_COLORS = [(40, 40, 230), (40, 180, 40), (230, 120, 20), (0, 150, 255)]  # BGR

MIN_TAGS_DEFAULT = 4  # per board view; 4 tags = 16 point pairs
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def tag_cell(tag_id):
    """(board, col, row) of a tag, row counted DOWN the sheet; None if not on a board."""
    if not 0 <= tag_id < NUM_BOARDS * TAGS_PER_BOARD:
        return None
    board, local = divmod(tag_id, TAGS_PER_BOARD)
    row_up, col = divmod(local, GRID_COLS)
    return board, col, GRID_ROWS - 1 - row_up


def cell_tag(board, col, row):
    return board * TAGS_PER_BOARD + (GRID_ROWS - 1 - row) * GRID_COLS + col


def tag_plane_corners(tag_id):
    """Board-plane corners in tag-edge units, x right and y down the sheet, origin at the
    top-left tag's outer corner. Detector order: bottom-left, bottom-right, top-right,
    top-left of the upright tag (checked against the rendered PNGs)."""
    _, col, row = tag_cell(tag_id)
    x, y = col * PITCH, row * PITCH
    return np.array([[x, y + 1], [x + 1, y + 1], [x + 1, y], [x, y]], np.float64)


def tag_world_corners(tag_id):
    """(4, 3) float32 board-frame corners in metres, Z = 0, in detector order."""
    xy = tag_plane_corners(tag_id) * TAG_SIZE
    return np.column_stack([xy, np.zeros(4)]).astype(np.float32)


# Unit-square corners of one tag, same order, for per-tag rectification.
UNIT_CORNERS = np.array([[0, 1], [1, 1], [1, 0], [0, 0]], np.float64)


# ── Camera prior ─────────────────────────────────────────────────────────────


class Camera:
    """Maps between distorted image pixels and undistorted (pinhole) pixels."""

    def __init__(self, K, D=None, fisheye=False):
        self.K = np.asarray(K, np.float64)
        self.D = None if D is None or not np.any(D) else np.asarray(D, np.float64)
        self.fisheye = fisheye

    def undistort(self, pts):
        pts = np.asarray(pts, np.float64).reshape(-1, 1, 2)
        if self.D is None:
            return pts.reshape(-1, 2)
        if self.fisheye:
            return cv2.fisheye.undistortPoints(pts, self.K, self.D, P=self.K).reshape(-1, 2)
        return cv2.undistortPoints(pts, self.K, self.D, P=self.K).reshape(-1, 2)

    def distort(self, pts):
        pts = np.asarray(pts, np.float64).reshape(-1, 2)
        if self.D is None:
            return pts
        ray = np.column_stack([(pts - self.K[:2, 2]) / np.diag(self.K)[:2], np.ones(len(pts))])
        zero = np.zeros(3)
        if self.fisheye:
            out, _ = cv2.fisheye.projectPoints(ray.reshape(-1, 1, 3), zero, zero, self.K, self.D)
        else:
            out, _ = cv2.projectPoints(ray, zero, zero, self.K, self.D)
        return out.reshape(-1, 2)


def exif_camera(path, width, height):
    """Pinhole guess from the EXIF 35 mm-equivalent focal length (diagonal based).

    Phone JPEGs have lens distortion corrected in the ISP, so zero distortion is a fair
    prior. Without EXIF, assumes a ~70 degree diagonal field of view. Only local
    homographies are built from it, so a rough focal length is harmless.
    """
    f35 = None
    try:
        f35 = Image.open(path).getexif().get_ifd(0x8769).get(0xA405)  # FocalLengthIn35mmFilm
    except Exception:
        pass
    f = (f35 / 43.27 if f35 else 0.71) * np.hypot(width, height)
    return Camera([[f, 0, (width - 1) / 2], [0, f, (height - 1) / 2], [0, 0, 1]])


# ── Geometry helpers ─────────────────────────────────────────────────────────


def apply_h(H, pts):
    pts = np.asarray(pts, np.float64).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(pts, H).reshape(-1, 2)


def edge_len(corners):
    return float(np.mean(np.linalg.norm(corners - np.roll(corners, 1, axis=0), axis=1)))


def in_frame(corners, width, height, margin):
    return bool(
        (corners[:, 0] >= margin).all()
        and (corners[:, 0] < width - margin).all()
        and (corners[:, 1] >= margin).all()
        and (corners[:, 1] < height - margin).all()
    )


def sample_patch(pyramid, cam, H_plane, plane_origin, plane_size, px_per_unit):
    """Warp a board-plane rectangle into a fronto-parallel patch.

    H_plane maps plane coords to undistorted pixels; patch pixel (u, v) sits at plane
    plane_origin + (u, v) / px_per_unit. Samples the pyramid level nearest the patch's
    resolution without going below it, so large tags are not aliased.
    Returns (patch, patch->plane as a 3x3).
    """
    w, h = (int(round(s * px_per_unit)) for s in plane_size)
    u, v = np.meshgrid(np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64))
    plane = np.column_stack([u.ravel(), v.ravel()]) / px_per_unit + plane_origin
    src = cam.distort(apply_h(H_plane, plane)).reshape(h, w, 2)
    span = np.linalg.norm(src[h // 2, -1] - src[h // 2, 0])  # image px across the patch
    level = int(np.clip(np.floor(np.log2(max(span / max(w - 1, 1), 1.0))), 0, len(pyramid) - 1))
    src = (src + 0.5) / 2.0**level - 0.5
    maps = src.astype(np.float32)
    patch = cv2.remap(
        pyramid[level],
        maps[..., 0],
        maps[..., 1],
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=255,
    )
    to_plane = np.array(
        [[1 / px_per_unit, 0, plane_origin[0]], [0, 1 / px_per_unit, plane_origin[1]], [0, 0, 1]]
    )
    return patch, to_plane


# ── Edge-line corner refinement ──────────────────────────────────────────────

REFINE_MARGIN = 0.2  # rectified patch extends this far past the tag (gap is 0.25)
REFINE_BAND = 0.1  # search half-width across an edge; the black border is 0.125


def _edge_points(patch, px, m):
    """Sub-pixel points on the TOP outer edge of a rectified tag patch.

    The tag occupies [m, m + px] in both axes. Going down a column the intensity drops
    from white paper to the black border: take the most negative vertical gradient within
    the band around y = m and refine it with a parabola.
    """
    band = max(2, int(round(REFINE_BAND * px)))
    lo = m - band
    cols = np.arange(int(m + 0.1 * px), int(m + 0.9 * px))
    grad = np.diff(patch[lo : m + band + 1, cols].astype(np.float32), axis=0)
    i = np.argmin(grad, axis=0)  # grad[i] sits at y = lo + i + 0.5
    k = np.arange(len(cols))
    a = grad[np.clip(i - 1, 0, None), k]
    b = grad[i, k]
    c = grad[np.clip(i + 1, None, grad.shape[0] - 1), k]
    denom = a - 2 * b + c
    off = np.where(np.abs(denom) > 1e-6, 0.5 * (a - c) / np.where(denom == 0, 1, denom), 0)
    ok = (i > 0) & (i < grad.shape[0] - 1) & (np.abs(off) < 1)
    # A real edge is a strong step; flat or noisy columns are dropped.
    if ok.any():
        ok &= -b > 0.25 * np.median(-b[ok])
    return np.column_stack([cols[ok].astype(np.float64), (lo + i + 0.5 + off)[ok]])


def refine_tag(pyramid, cam, corners_dist, iterations=2):
    """Refit a tag's outer corners as intersections of sub-pixel edge lines.

    Works in a patch rectified by the tag's own corners, so the edges are axis-aligned
    and every pixel column/row gives one edge sample. A homography keeps lines straight,
    so lines here are lines in the undistorted image. Returns refined distorted-image
    corners, or None if the fit is unreliable.
    """
    und = cam.undistort(corners_dist)
    px = int(np.clip(2 * edge_len(und), 48, 400))
    m = int(round(REFINE_MARGIN * px))
    for _ in range(iterations):
        H = cv2.getPerspectiveTransform(UNIT_CORNERS.astype(np.float32), und.astype(np.float32))
        patch, to_unit = sample_patch(
            pyramid,
            cam,
            H,
            (-REFINE_MARGIN, -REFINE_MARGIN),
            (1 + 2 * REFINE_MARGIN, 1 + 2 * REFINE_MARGIN),
            px,
        )
        size = patch.shape[0]
        lines = []
        # Rotate so each edge in turn is the top one: top, right, bottom, left.
        for k in range(4):
            pts = _edge_points(np.rot90(patch, k), px, m)
            if len(pts) < 0.3 * px:
                return None
            x, y = pts[:, 0], pts[:, 1]
            for _ in range(k):  # undo each 90 degree rotation
                x, y = size - 1 - y, x
            vx, vy, x0, y0 = cv2.fitLine(
                np.column_stack([x, y]).astype(np.float32), cv2.DIST_HUBER, 0, 0.01, 0.01
            ).ravel()
            lines.append(np.array([vy, -vx, vx * y0 - vy * x0], np.float64))
        top, right, bottom, left = lines
        new = np.array(
            [
                np.cross(left, bottom),
                np.cross(right, bottom),
                np.cross(right, top),
                np.cross(left, top),
            ]
        )
        if np.any(np.abs(new[:, 2]) < 1e-9):
            return None
        new_und = apply_h(H, apply_h(to_unit, new[:, :2] / new[:, 2:]))
        if np.max(np.linalg.norm(new_und - und, axis=1)) > 0.1 * edge_len(und):
            return None
        und = new_und
    return cam.distort(und)


# ── Board fitting and prediction ─────────────────────────────────────────────


def local_homography(tags, board, col, row, cam, exclude=None, max_tags=8):
    """Plane->undistorted homography from the found tags nearest (col, row) on a board.

    Local rather than whole-board so curl in the sheet and leftover lens distortion only
    cost what they do across a few tags. Needs at least two tags within two cells.
    Returns (H, n_tags), or (None, 0).
    """
    near = []
    for tid in tags:
        b, c, r = tag_cell(tid)
        if b == board and tid != exclude:
            near.append((max(abs(c - col), abs(r - row)), abs(c - col) + abs(r - row), tid))
    near = sorted(near)[:max_tags]
    if len(near) < 2 or near[0][0] > 2:
        return None, 0
    plane = np.vstack([tag_plane_corners(t) for *_, t in near])
    img = np.vstack([cam.undistort(tags[t]["corners"]) for *_, t in near])
    H, _ = cv2.findHomography(plane, img, 0)
    return H, len(near)


def predict_tag(tags, tid, cam):
    """Distorted-image corners where a tag should be, from its board neighbours."""
    board, col, row = tag_cell(tid)
    H, _ = local_homography(tags, board, col, row, cam)
    if H is None:
        return None, None
    return H, cam.distort(apply_h(H, tag_plane_corners(tid)))


def occluded(pred, tags, board):
    """True if a predicted tag sits under a tag found on another board.

    Sheets are laid down overlapping, so part of one board is routinely covered by
    another; such cells are neither recoverable nor missing.
    """
    center = tuple(pred.mean(axis=0))
    for tid, t in tags.items():
        if tag_cell(tid)[0] == board:
            continue
        quad = t["corners"].astype(np.float32)
        if (
            cv2.pointPolygonTest(quad, center, False) >= 0
            or cv2.pointPolygonTest(pred.astype(np.float32), tuple(quad.mean(axis=0)), False) >= 0
        ):
            return True
    return False


def visible(pred, tags, board, shape):
    """A predicted tag fully inside the frame, big enough to read, and not covered."""
    return (
        in_frame(pred, shape[1], shape[0], 2)
        and edge_len(pred) >= 8
        and not occluded(pred, tags, board)
    )


def loo_residuals(tags, cam):
    """Leave-one-out residual of each tag vs. the prediction from its neighbours, as a
    fraction of its edge. Corner noise plus sheet curl; > 5% is flagged in debug images.
    """
    out = {}
    for tid, t in tags.items():
        b, c, r = tag_cell(tid)
        H, n = local_homography(tags, b, c, r, cam, exclude=tid)
        if H is None or n < 3:
            continue
        und = cam.undistort(t["corners"])
        err = np.sqrt(np.mean(np.sum((apply_h(H, tag_plane_corners(tid)) - und) ** 2, axis=1)))
        out[tid] = float(err / edge_len(und))
    return out


PATCH_MARGIN = 0.45  # predicted patch extends this far past the tag on every side
PATCH_TAG_PX = 160  # tag edge in the predicted patch (20 px per cell)


def recover_board(det, shape, pyramid, cam, tags, board, max_hamming):
    """Grow a board's detections into neighbouring cells via rectified patches."""
    tried = set()
    while True:
        found = {tag_cell(t)[1:] for t in tags if tag_cell(t)[0] == board}
        frontier = {
            (c + dc, r + dr)
            for c, r in found
            for dc in (-1, 0, 1)
            for dr in (-1, 0, 1)
            if 0 <= c + dc < GRID_COLS and 0 <= r + dr < GRID_ROWS
        }
        frontier -= found | tried
        if not frontier:
            return
        added = 0
        for c, r in sorted(frontier):
            tried.add((c, r))
            tid = cell_tag(board, c, r)
            H, pred = predict_tag(tags, tid, cam)
            if H is None or not visible(pred, tags, board, shape):
                continue
            x0, y0 = tag_plane_corners(tid)[3]
            patch, to_plane = sample_patch(
                pyramid,
                cam,
                H,
                (x0 - PATCH_MARGIN, y0 - PATCH_MARGIN),
                (1 + 2 * PATCH_MARGIN, 1 + 2 * PATCH_MARGIN),
                PATCH_TAG_PX,
            )
            hits = [d for d in det.detect(patch) if d.tag_id == tid and d.hamming <= max_hamming]
            if not hits:
                continue
            corners = cam.distort(apply_h(H, apply_h(to_plane, hits[0].corners)))
            # The decode must land where the board says it is.
            if np.max(np.linalg.norm(corners - pred, axis=1)) > 0.35 * edge_len(pred):
                continue
            tags[tid] = {"corners": corners, "source": 2, "hamming": int(hits[0].hamming)}
            added += 1
        if not added:
            return


# ── Per-image pipeline (runs in worker processes) ────────────────────────────

_DETS = {}


def _init_worker():
    # One detector set per worker. Workers leave via os._exit, so the pupil-apriltags
    # double-free in Detector.__del__ (see calibrate.release_detectors) never runs.
    cv2.setNumThreads(1)

    def mk(dec):
        return Detector(
            families=TAG_FAMILY,
            nthreads=1,
            quad_decimate=dec,
            refine_edges=1,
            decode_sharpening=0.25,
        )

    _DETS["fine"] = mk(1.0)
    _DETS["coarse"] = mk(2.0)


def detect_image(job):
    """Detect, recover and refine all board tags in one image; returns a JSON-able dict."""
    path, opts = job
    t0 = time.time()
    gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE | cv2.IMREAD_IGNORE_ORIENTATION)
    if gray is None:
        return {"image": path.name, "error": "could not read"}
    h, w = gray.shape
    prior = opts["prior"]
    if prior is not None and prior["size"] == (w, h):
        cam = Camera(prior["K"], prior["D"], prior["fisheye"])
    else:
        cam = exif_camera(path, w, h)
    pyramid = [gray]
    while min(pyramid[-1].shape) > 64:
        pyramid.append(cv2.pyrDown(pyramid[-1]))

    # Pass 1: exact decodes only. Earlier runs win a tag id, so the finest one supplies
    # the corners; the coarse ones only add tags, and refinement sharpens them.
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(16, 16)).apply(gray)
    runs = [(gray, "fine", 1), (gray, "coarse", 1), (clahe, "coarse", 1)]
    runs += [(pyramid[lv], "fine", 2**lv) for lv in (1, 2) if lv < len(pyramid)]
    tags = {}
    for img, det, scale in runs:
        for d in _DETS[det].detect(img):
            if d.hamming == 0 and tag_cell(d.tag_id) and d.tag_id not in tags:
                corners = (d.corners.astype(np.float64) + 0.5) * scale - 0.5
                tags[d.tag_id] = {"corners": corners, "source": 1, "hamming": 0}
    n_pass1 = len(tags)

    # Pass 2: board-guided recovery.
    if opts["pass2"]:
        for board in range(NUM_BOARDS):
            if sum(tag_cell(t)[0] == board for t in tags) >= 2:
                recover_board(
                    _DETS["fine"], gray.shape, pyramid, cam, tags, board, opts["max_hamming"]
                )

    refine_fail = 0
    if opts["refine"]:
        for t in tags.values():
            new = refine_tag(pyramid, cam, t["corners"])
            if new is None:
                refine_fail += 1
            else:
                t["corners"] = new

    # Cells that should be readable on a found board but were not detected, for review.
    missing = {}
    for board in range(NUM_BOARDS):
        if sum(tag_cell(t)[0] == board for t in tags) < 2:
            continue
        for tid in range(board * TAGS_PER_BOARD, (board + 1) * TAGS_PER_BOARD):
            if tid not in tags:
                H, pred = predict_tag(tags, tid, cam)
                if H is not None and visible(pred, tags, board, gray.shape):
                    missing[tid] = pred.tolist()

    return {
        "image": path.name,
        "width": w,
        "height": h,
        "tags": {
            int(t): {
                "corners": v["corners"].tolist(),
                "source": v["source"],
                "hamming": v["hamming"],
            }
            for t, v in sorted(tags.items())
        },
        "n_pass1": n_pass1,
        "n_pass2": len(tags) - n_pass1,
        "missing": missing,
        "loo": loo_residuals(tags, cam),
        "refine_fail": refine_fail,
        "seconds": time.time() - t0,
    }


def render_debug(job):
    """Write the debug overlay for one image (worker side, so it runs in parallel)."""
    path, result, out_dir = job
    img = draw_debug(
        cv2.imread(str(path), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION), result
    )
    cv2.imwrite(str(out_dir / path.name), img, [cv2.IMWRITE_JPEG_QUALITY, 88])


# ── Debug rendering ──────────────────────────────────────────────────────────


def _dashed(img, pts, color, thick, dash):
    for a, b in zip(pts, np.roll(pts, -1, axis=0)):
        n = max(1, int(np.linalg.norm(b - a) / dash))
        for i in range(0, n, 2):
            p = a + (b - a) * i / n
            q = a + (b - a) * min(i + 1, n) / n
            cv2.line(
                img,
                tuple(np.round(p).astype(int)),
                tuple(np.round(q).astype(int)),
                color,
                thick,
                cv2.LINE_AA,
            )


def _label(img, text, center, scale, color, thick):
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
    org = (int(center[0] - tw / 2), int(center[1] + th / 2))
    cv2.putText(img, text, org, font, scale, (0, 0, 0), thick + max(2, thick), cv2.LINE_AA)
    cv2.putText(img, text, org, font, scale, color, thick, cv2.LINE_AA)


def draw_debug(bgr, result):
    """Overlay one image's detections.

    Outline colour is the board (A red, B green, C blue, D orange): solid = pass 1,
    dashed = recovered in pass 2, grey dashed = should be readable but was not found.
    The id is at the tag centre (with hN if decoded with N bit errors), a dot marks
    detector corner 0 (bottom-left of the upright tag), and a red % under the id flags
    a tag that disagrees with its neighbours by more than 5% of its edge.
    """
    img = (bgr * 0.6 + 40).astype(np.uint8)  # dim the photo so overlays stand out
    base = max(img.shape[:2]) / 4080
    for pred in result["missing"].values():
        pred = np.array(pred)
        _dashed(img, pred, (170, 170, 170), max(1, int(2 * base)), max(6, edge_len(pred) / 8))
    for tid, t in result["tags"].items():
        tid = int(tid)
        color = BOARD_COLORS[tag_cell(tid)[0]]
        c = np.array(t["corners"])
        e = edge_len(c)
        thick = int(np.clip(e / 40, 1, 6 * base + 1))
        if t["source"] == 1:
            cv2.polylines(img, [np.round(c).astype(np.int32)], True, color, thick, cv2.LINE_AA)
        else:
            _dashed(img, c, color, thick + 1, max(4, e / 10))
        cv2.circle(img, tuple(np.round(c[0]).astype(int)), thick + 2, color, -1, cv2.LINE_AA)
        scale = float(np.clip(e / 110, 0.35, 3.0))
        text = str(tid) + ("" if t["hamming"] == 0 else f"h{t['hamming']}")
        _label(img, text, c.mean(axis=0), scale, (255, 255, 255), max(1, int(scale * 2)))
        rel = result["loo"].get(tid, result["loo"].get(str(tid), 0.0))
        if rel > 0.05:
            _label(
                img,
                f"{100 * rel:.0f}%",
                c.mean(axis=0) + [0, e * 0.3],
                scale * 0.7,
                (60, 60, 255),
                max(1, int(scale * 1.5)),
            )

    counts = board_counts(result)
    header = (
        f"{result['image']}   tags {len(result['tags'])} (pass2 +{result['n_pass2']})   "
        f"missing {len(result['missing'])}   "
        + "  ".join(f"{BOARD_NAMES[b]}:{counts[b]}" for b in range(NUM_BOARDS))
    )
    s, t = 1.4 * base, max(1, int(3 * base))
    cv2.rectangle(img, (0, 0), (img.shape[1], int(70 * base)), (0, 0, 0), -1)
    cv2.putText(
        img,
        header,
        (int(20 * base), int(50 * base)),
        cv2.FONT_HERSHEY_SIMPLEX,
        s,
        (255, 255, 255),
        t,
        cv2.LINE_AA,
    )
    for b in range(NUM_BOARDS):
        x0 = img.shape[1] - int((NUM_BOARDS - b) * 160 * base)
        cv2.rectangle(
            img, (x0, int(15 * base)), (x0 + int(140 * base), int(60 * base)), BOARD_COLORS[b], -1
        )
        cv2.putText(
            img,
            BOARD_NAMES[b],
            (x0 + int(55 * base), int(52 * base)),
            cv2.FONT_HERSHEY_SIMPLEX,
            s,
            (0, 0, 0),
            t,
        )
    return img


# ── Detection over a folder ──────────────────────────────────────────────────


def board_counts(result):
    return [sum(int(t) // TAGS_PER_BOARD == b for t in result["tags"]) for b in range(NUM_BOARDS)]


def detect_folder(image_dir: Path, args, debug_dir: Path | None, prior):
    """Run detection over every image in parallel; print a per-image summary."""
    paths = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    if not paths:
        sys.exit(f"No images found in {image_dir}")
    opts = {
        "prior": prior,
        "max_hamming": args.max_hamming,
        "pass2": not args.no_pass2,
        "refine": not args.no_refine,
    }
    print(f"Detecting tags in {len(paths)} images with {args.jobs} worker(s) …")
    with Pool(min(args.jobs, len(paths)), initializer=_init_worker) as pool:
        results = pool.map(detect_image, [(p, opts) for p in paths])
        if debug_dir is not None:
            debug_dir.mkdir(parents=True, exist_ok=True)
            pool.map(
                render_debug,
                [(p, r, debug_dir) for p, r in zip(paths, results) if "error" not in r],
            )
            with open(debug_dir / "detections.json", "w") as f:
                json.dump({"tag_size": TAG_SIZE, "images": results}, f)

    print(
        f"\n  {'image':34} {'tags':>4} {'+p2':>4} {'miss':>4}  "
        + " ".join(f"{n:>3}" for n in BOARD_NAMES)
        + f"  {'sec':>4}"
    )
    for r in results:
        if "error" in r:
            print(f"  {r['image'][:34]:34} {r['error']}")
            continue
        print(
            f"  {r['image'][:34]:34} {len(r['tags']):>4} {r['n_pass2']:>4} "
            f"{len(r['missing']):>4}  "
            + " ".join(f"{c:>3}" for c in board_counts(r))
            + f"  {r['seconds']:>4.1f}"
        )
    ok = [r for r in results if "error" not in r]
    found = sum(len(r["tags"]) for r in ok)
    miss = sum(len(r["missing"]) for r in ok)
    print(
        f"\n  {found} tags ({sum(r['n_pass2'] for r in ok)} recovered in pass 2), "
        f"{miss} readable-looking tags not found "
        f"({found / max(found + miss, 1):.1%} of expected)"
    )
    if debug_dir is not None:
        print(f"  debug images and detections.json in {debug_dir}")
    if args.visualize:
        for p, r in zip(paths, results):
            if "error" in r:
                continue
            img = draw_debug(
                cv2.imread(str(p), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION), r
            )
            f = 1200 / max(img.shape[:2])
            cv2.imshow(
                "detections", cv2.resize(img, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
            )
            cv2.waitKey(300)
        cv2.destroyAllWindows()
    return paths, results


def build_views(paths, results, min_tags):
    """One calibration view per (image, board) with >= min_tags tags.

    Only the most common image size is used: a phone that bakes its rotation into the
    pixels gives portrait frames, which are a differently oriented camera.
    """
    sizes = [(r["width"], r["height"]) for r in results if "error" not in r]
    if not sizes:
        return [], [], [], None
    image_size = max(set(sizes), key=sizes.count)
    all_obj, all_img, names = [], [], []
    for path, r in zip(paths, results):
        if "error" in r:
            print(f"  [skip] {r['image']}: {r['error']}")
            continue
        if (r["width"], r["height"]) != image_size:
            print(
                f"  [skip] {r['image']}: {r['width']}x{r['height']}, calibrating "
                f"{image_size[0]}x{image_size[1]}"
            )
            continue
        for b in range(NUM_BOARDS):
            ids = [int(t) for t in r["tags"] if int(t) // TAGS_PER_BOARD == b]
            if not ids:
                continue
            if len(ids) < min_tags:
                print(
                    f"  [skip] {path.name} board {BOARD_NAMES[b]}: {len(ids)} tags "
                    f"(need {min_tags})"
                )
                continue
            all_obj.append(np.vstack([tag_world_corners(t) for t in ids]))
            all_img.append(np.vstack([r["tags"][t]["corners"] for t in ids]).astype(np.float32))
            # calibrate_robust and print_view_errors name views by .name
            names.append(Path(f"{path.name} board {BOARD_NAMES[b]}"))
    return all_obj, all_img, names, image_size


# ── Per-camera calibration run ───────────────────────────────────────────────


def load_prior(path: Path | None, model: str):
    """Prior intrinsics for undistorting board geometry, from a camera_info YAML."""
    if path is None:
        return None
    if not path.exists():
        sys.exit(f"--initial-guess file not found: {path}")
    with open(path) as f:
        data = yaml.safe_load(f)
    K, D = load_prior_calibration(path)
    print(f"Initial-guess intrinsics loaded from {path}")
    return {
        "K": K,
        "D": D.ravel(),
        "size": (data.get("image_width"), data.get("image_height")),
        "fisheye": (data.get("distortion_model") or model) in ("fisheye", "equidistant"),
    }


def run_camera(cam: str, image_dir: Path, device_dir: Path, args):
    """Calibrate one camera and write its main + debug camera_info YAMLs."""
    spec = CAMERAS[cam]
    main_path, debug_path = output_paths(device_dir, cam)

    print(f"\n{'=' * 70}")
    print(f"Camera       : {cam}  ({spec['model']})")
    print(f"Image folder : {image_dir}")
    print(f"Output       : {main_path}")
    print(f"               {debug_path}\n")

    if not image_dir.is_dir():
        sys.exit(f"Image folder not found: {image_dir}")

    debug_dir = image_dir / "debug_images" if args.debug_images else None
    prior = load_prior(args.initial_guess, spec["model"])
    paths, results = detect_folder(image_dir, args, debug_dir, prior)

    print()
    all_obj, all_img, names, image_size = build_views(paths, results, args.min_tags)
    print(f"\nUsable board views: {len(names)} from {len(paths)} images")
    if len(names) < 3:
        sys.exit("Need at least 3 usable board views for calibration.")

    print("Running calibration …")
    calib_fn, project = (
        (calibrate_fisheye, _project_fisheye)
        if spec["model"] == "fisheye"
        else (calibrate_zoom, _project_zoom)
    )
    K, D, _rvecs, _tvecs, rms, _o, _i, kept, errs = calibrate_robust(
        all_obj,
        all_img,
        names,
        image_size,
        calib_fn,
        project,
    )
    print(f"\nKept {len(kept)} / {len(names)} views after rejection")
    print(f"Overall RMS reprojection error: {rms:.4f} px")
    print_view_errors(kept, errs, spec["model"])

    write_yaml(camera_info(K, D, image_size, spec["main_name"], spec["model"], rms), main_path)
    debug_size = spec["debug_size"]
    write_yaml(
        camera_info(
            _scale_K(K, image_size, debug_size), D, debug_size, spec["debug_name"], spec["model"]
        ),
        debug_path,
    )


# ── Entry point ──────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Multi-board AprilGrid camera intrinsic calibration → device config layout"
    )
    parser.add_argument("--serial", required=True, help="Device serial, e.g. vss_00000041")
    parser.add_argument(
        "--context",
        type=Path,
        default=None,
        metavar="IMAGE_DIR",
        help="Folder of context-camera images (fisheye model)",
    )
    parser.add_argument(
        "--zoom",
        type=Path,
        default=None,
        metavar="IMAGE_DIR",
        help="Folder of zoom-camera images (plumb_bob model)",
    )

    parser.add_argument(
        "--devices-root",
        type=Path,
        default=Path(__file__).resolve().parent / DEVICES_DIRNAME,
        help="Root of the devices/ tree (default: ./devices next to this script)",
    )
    parser.add_argument(
        "-y",
        "--yes",
        action="store_true",
        help="Overwrite existing calibration files without asking",
    )
    parser.add_argument(
        "--initial-guess",
        type=Path,
        default=None,
        help="Path to an intrinsics YAML used as a prior: board geometry is "
        "undistorted with it, so predicted tags land where they are. Pass "
        "a previous calibration here to refine it.",
    )
    parser.add_argument(
        "--min-tags",
        type=int,
        default=MIN_TAGS_DEFAULT,
        help=f"Min detected tags per board view (default: {MIN_TAGS_DEFAULT})",
    )
    parser.add_argument(
        "--visualize", action="store_true", help="Display tag detections after processing"
    )
    parser.add_argument(
        "--debug-images",
        action="store_true",
        help="Save annotated debug images to debug_images/ in the image folder",
    )

    parser.add_argument(
        "--jobs",
        type=int,
        default=min(4, os.cpu_count() or 1),
        help="Detection worker processes, each single-threaded (default: 4)",
    )
    parser.add_argument(
        "--max-hamming",
        type=int,
        default=2,
        help="Bit errors accepted in pass 2, for the expected id only",
    )
    parser.add_argument(
        "--no-pass2", action="store_true", help="Skip board-guided recovery of missed tags"
    )
    parser.add_argument(
        "--no-refine",
        action="store_true",
        help="Keep detector corners instead of sub-pixel edge-line corners",
    )
    args = parser.parse_args()

    requested = [
        (cam, d) for cam, d in (("context", args.context), ("zoom", args.zoom)) if d is not None
    ]
    if not requested:
        parser.error("pass --context IMAGE_DIR and/or --zoom IMAGE_DIR")
    if args.initial_guess is not None and len(requested) > 1:
        sys.exit("--initial-guess describes one camera; pass only --context or --zoom with it.")

    serial = normalize_serial(args.serial)
    device_dir = args.devices_root / serial

    print(f"\nDevice       : {serial}")
    print(f"Device folder: {device_dir / CAMERA_SUBDIR}")

    existing = [p for cam, _ in requested for p in output_paths(device_dir, cam) if p.exists()]
    if not confirm_overwrite(existing, args.yes):
        sys.exit("Aborted — existing calibration left untouched.")

    (device_dir / CAMERA_SUBDIR).mkdir(parents=True, exist_ok=True)

    for cam, image_dir in requested:
        run_camera(cam, image_dir, device_dir, args)

    print(f"\nDone. Calibration written to {device_dir / CAMERA_SUBDIR}")


if __name__ == "__main__":
    main()
