# `aprilgrid_detect.py` — intrinsics from the four-board AprilGrid set

Drop-in replacement for [`calibrate.py`](SETUP.md#calibratepy--camera-intrinsics-from-an-aprilgrid)
for the multi-board target: four 8×8 tag36h11 sheets that can lie anywhere in the
image, each at its own pose. Same command line, same camera_info YAMLs in the
device-config layout:

```
devices/<serial>/ip_camera_processing_cpp/
    context_camera_info.yaml        zoom_camera_info.yaml
    debug_context_camera_info.yaml  debug_zoom_camera_info.yaml
```

Like `calibrate.py` it is a standalone script, not part of the installed package, and
needs `pupil-apriltags` (see [SETUP.md](SETUP.md)).

## Usage

```bash
source .venv/bin/activate

# Fisheye / wide-angle lens (equidistant model)
python aprilgrid_detect.py --serial vss_00000041 --context path/to/context_images

# Standard / zoom lens (plumb_bob model)
python aprilgrid_detect.py --serial vss_00000041 --zoom path/to/zoom_images

# Both in one run, with annotated detections for review
python aprilgrid_detect.py --serial vss_00000041 \
    --context path/to/context_images --zoom path/to/zoom_images --debug-images
```

| Flag | Purpose |
| --- | --- |
| `--serial SERIAL` | Device serial (required). `41` and `00000041` expand to `vss_00000041`. |
| `--context DIR` / `--zoom DIR` | Image folders; fisheye and plumb_bob model respectively |
| `--devices-root PATH` | Root of the `devices/` tree (default: `devices/` next to the script) |
| `-y`, `--yes` | Overwrite existing calibration files without asking |
| `--initial-guess PATH` | Prior intrinsics YAML. Board geometry is undistorted with it, so tag predictions land in the right place. Use it to refine an earlier calibration; most useful for fisheye. |
| `--min-tags N` | Minimum tags for a board to count as a view (default 4) |
| `--debug-images` | Write annotated images and `detections.json` to `debug_images/` in the image folder |
| `--visualize` | Show the annotated images after detection |
| `--jobs N` | Detection worker processes, each single-threaded (default 4) |
| `--max-hamming N` | Bit errors tolerated when re-reading a predicted tag (default 2) |
| `--no-pass2`, `--no-refine` | Turn off board-guided recovery / sub-pixel edge refinement |

## The board

`make_aprilgrid.py` renders the four sheets into `aprilgrid_23in_8x8/`:

| Board | Tag ids | Colour in debug images |
| --- | --- | --- |
| A | 0–63 | red |
| B | 64–127 | green |
| C | 128–191 | blue |
| D | 192–255 | orange |

Each sheet is 8×8 tags, ids row-major from the bottom-left, tag spacing 0.25 of the
tag edge, no checker squares. The script needs no information about where the sheets
are relative to each other: every sheet seen in an image is its own calibration view
with its own pose. A sheet partly covering another is fine. The covered tags are
recognised as occluded rather than missed.

The tag size (`TAG_SIZE`, 56.2 mm as rendered) does not affect the intrinsics, only the
scale of the solved board poses.

## Capturing images

What limits accuracy is the board, not the detector:

- **Keep the sheets flat.** Mount them on something rigid (foam board, Dibond, glass).
  Curl and lifted edges show up directly as reprojection error. On a set taken with
  loose overlapping sheets, a single sheet fit as a perfect plane still left ~1.8 px.
- **Keep the camera in one orientation.** Only the most common image size is
  calibrated. A phone that bakes rotation into the pixels produces portrait and
  landscape frames that are effectively different cameras, so the minority size is
  skipped.
- **Cover the frame, corners included,** at a range of tilts, but avoid grazing angles
  and very close shots. Far rows at a grazing angle are out of focus, and close-ups
  see only a few tags.
- **Watch for glare.** Washed-out tags are often lost.

## How detection works

Per image, in parallel worker processes:

1. **Pass 1:** pupil-apriltags over the whole frame at decimation 1 and 2, on a
   contrast-normalised (CLAHE) copy for glare, and on 2× and 4× downscales for large,
   defocused tags close to the camera. Only exact (hamming 0) decodes are kept.
2. **Pass 2:** for each board, every cell next to a found tag is predicted with a
   homography of its found neighbours. The region is warped to a fronto-parallel
   patch, which flattens grazing views and enlarges small tags, and decoded there. Only
   the id that belongs in that cell is accepted, so a few bit errors can be tolerated
   safely. This repeats until the board stops growing.
3. **Refinement:** each tag is rectified through its own corners. Its four outer edges
   are fitted as sub-pixel lines, which are intersected for the corners.

All board geometry is worked in undistorted pixels, from `--initial-guess` or else a
pinhole guess from the EXIF focal length, so a flat sheet is exactly a homography.
Tags cut by the image border are not used.

Calibration then runs per camera with `calibrate.py`'s robust loop, which drops
mislocalised tags and bad views, and its fisheye/plumb_bob solvers and YAML writer.

## Reading the debug images

With `--debug-images`, each image gets an overlay in `debug_images/`:

- **Outline colour** is the board (table above). **Solid** lines are found in pass 1,
  **dashed** ones were recovered in pass 2.
- **Grey dashed** outlines are tags the board says are fully in frame and readable
  but that were not found.
- **White number:** the tag id, with `hN` appended if it was read with N bit errors.
  The **dot** marks detector corner 0, the bottom-left of the upright tag.
- **Red %:** the tag disagrees with its neighbours by more than 5% of its edge. This
  usually means a curled sheet, occasionally a bad corner.

The header shows the total, pass-2 additions, missing count and per-board counts. The
console prints the same table for the whole folder.
