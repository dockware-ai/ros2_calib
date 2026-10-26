# Setup & Running

How to bring this repo up from a fresh clone on a new machine, and how to run each
of the tools it ships.

Everything runs from a single virtualenv at `.venv/` in the repo root. There is no
ROS 2 installation required to *run* the calibration tools — bags are read with the
pure-Python `rosbags` library. ROS 2 (or Docker) is only needed to *capture* bags
with `capture_bag.sh`.

## 1. System prerequisites

- Python 3.10 or newer (`python3 --version`)
- `python3-venv` and the Qt/OpenGL runtime libraries the GUI needs:

```bash
sudo apt update
sudo apt install -y python3-venv python3-dev \
    libxcb-cursor0 libxcb-xinerama0 libxkbcommon-x11-0 libgl1 libglib2.0-0
```

The `libxcb-*` / `libgl1` packages are what PySide6 and Open3D link against at
runtime. Without them the app fails at startup with
`qt.qpa.plugin: Could not load the Qt platform plugin "xcb"`.

A graphical session (X11 or Wayland with XWayland) is required — both calibration
workflows are interactive GUIs. Over SSH, use `ssh -X` or a remote desktop.

## 2. Create the virtualenv and install

From the repo root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

`-e` (editable) installs the package as a link to the source tree, so edits to
`ros2_calib/*.py` take effect without reinstalling. Drop the `-e` for a plain
install.

This single command installs **all** runtime dependencies, including the ones
`calibrate.py` needs (`pupil-apriltags`) and the ones the GUI widgets need
(`matplotlib`). Both are declared in `pyproject.toml` — nothing should need to be
`pip install`ed by hand.

Optional dev tooling (linter, packaging):

```bash
python -m pip install -e ".[dev]"
```

### Verify the install

```bash
source .venv/bin/activate

# calibrate.py and its AprilTag dependency
python -c "import pupil_apriltags, cv2, yaml, numpy; print('calibrate deps OK')"
python calibrate.py --help

# GUI stack (offscreen so it works without a display)
QT_QPA_PLATFORM=offscreen python -c "import ros2_calib.main; print('GUI imports OK')"
```

All three should succeed before you try the GUI for real.

## 3. Using the venv day to day

```bash
source .venv/bin/activate    # activate — prompt gains a (.venv) prefix
deactivate                   # leave it
```

You can skip activation by calling the interpreter directly, which is handy in
scripts and cron:

```bash
/path/to/ros2_calib/.venv/bin/python calibrate.py ...
/path/to/ros2_calib/.venv/bin/ros2_calib
```

**Do not** `pip install` into the system Python (`pip3 install --user ...`). That is
how the previous machine ended up with `pupil-apriltags` visible to the system
interpreter but missing from the venv — `calibrate.py` worked from a bare shell and
broke inside the venv. Always activate first, or use `.venv/bin/python -m pip`.

The `.venv/` directory is gitignored, so it must be created per machine. It is never
committed and never copied between machines (it contains absolute paths).

## 4. Running the tools

### `ros2_calib` — the GUI calibration tool

```bash
source .venv/bin/activate
ros2_calib
```

Then, in the app:

1. Pick the camera — **Context** or **Zoom**. Switching cameras after a bag is
   loaded re-processes it for that camera's image topic automatically.
2. Load a rosbag (`.mcap`) — the dialog opens in `bags/`.
3. Topics are fixed (`/context_camera/main` or `/zoom_camera/main`, plus
   `/livox/lidar` and the matching `camera_info`) and only displayed; the row turns
   red if the loaded bag is missing one. The ROS distro is fixed too — see
   `ROS_DISTRO` in `ros2_calib/main_window.py` if a bag ever needs a different one.
4. Set the initial transform / pick a synchronized frame.
5. Place correspondences. On the image view: **middle-click** starts one at that
   pixel, **left-click** picks (or unpicks) LiDAR points, **right-click** saves it
   with whatever is selected. The Add Correspondence / Confirm buttons still work
   the same way.
6. Export with **Export to Device** (the button in the calibration side panel does
   the same thing). The success message shows the absolute path written, and is
   only shown after the transform has been read back from the file.

The **Serial** field in the top bar decides which `devices/<serial>/` folder is
read and written. It fills in from the bag name when that carries a `vss_XXXXXXXX`
serial, turns red when a bag is loaded and no serial is known, and accepts a typed
one (`41`, `00000041` and `vss_41` expand to `vss_00000041`). A typed serial
overrides the bag name until a bag whose name carries one is loaded. A serial that
was only auto-filled from the previous bag is cleared when a bag without one is
loaded, so an export never lands in the previous device's folder.

Intrinsics are auto-loaded in this order: the device's own calibration at
`devices/<serial>/ip_camera_processing_cpp/<camera>_camera_info.yaml` (serial read
from the bag name), then the bag's `CameraInfo` topic, then the packaged default
file. It reloads whenever a bag is loaded or the camera is switched, and the
source label shows the full path of the file in use. The device file wins because `calibrate.py` wrote it for that exact unit,
while the bag's `CameraInfo` is often stale. The **Device File** button reloads it
on demand.

Exported transforms land in the device-config tree that mirrors
`vision-config/devices/`:

```
devices/<serial>/spatial_processing/static_transforms.yaml
```

The serial is read from the loaded bag path (`vss_00000029-zoom` → `vss_00000029`);
if no valid `vss_XXXXXXXX` serial is known the export asks for one (or for a file
to save into) instead of writing anywhere.
The export **merges** into `static_transforms.yaml`: only the key being exported
(`livox_to_context` or `livox_to_zoom`) is replaced, so calibrating the zoom camera
leaves the context transform — and anything else in the file — untouched.

### `calibrate.py` — camera intrinsics from an AprilGrid

Standalone script (not part of the installed package) that computes camera
intrinsics from a folder of AprilGrid (tag36h11) images.

Results are written straight into the device-config layout, mirroring
`vision-config/devices/`:

```
devices/<serial>/ip_camera_processing_cpp/
    context_camera_info.yaml        zoom_camera_info.yaml
    debug_context_camera_info.yaml  debug_zoom_camera_info.yaml
```

The `debug_*` files are the same intrinsics scaled to the low-resolution debug
stream (context 320×240, zoom 352×240); distortion coefficients are resolution
independent and carry over unchanged. The device folder is created if missing,
and if any of the target files already exist the script lists them and asks
before overwriting.

```bash
source .venv/bin/activate

# Fisheye / wide-angle lens (equidistant model)
python calibrate.py --serial vss_00000041 --context path/to/context_images

# Standard / zoom lens (plumb_bob model)
python calibrate.py --serial vss_00000041 --zoom path/to/zoom_images

# Both in one run — writes all four files
python calibrate.py --serial vss_00000041 \
    --context path/to/context_images --zoom path/to/zoom_images
```

Useful flags:

| Flag | Purpose |
| --- | --- |
| `--serial SERIAL` | Device serial (required). `41` and `00000041` are expanded to `vss_00000041`. |
| `--devices-root PATH` | Root of the `devices/` tree (default: `devices/` next to the script) |
| `-y`, `--yes` | Overwrite existing calibration files without asking |
| `--initial-guess PATH` | Prior intrinsics YAML; images are undistorted before detection so more tags are found. `--context` only. Use to refine an earlier calibration. |
| `--min-tags N` | Minimum tags per image to accept it (default 4) |
| `--visualize` | Show detections interactively |
| `--debug-images` | Dump annotated detection images |

Run `python calibrate.py --help` for the full list.

### `aprilgrid_detect.py` — intrinsics from the four-board AprilGrid

Same command line and outputs as `calibrate.py`, for the four 8×8 sheets in
`aprilgrid_23in_8x8/` laid out anywhere in the image:

```bash
python aprilgrid_detect.py --serial vss_00000041 --zoom path/to/zoom_images --debug-images
```

See [APRILGRID_DETECT.md](APRILGRID_DETECT.md) for capture tips, flags and how to read
the debug images.

### `capture_bag.sh` — record a bag from a sensor suite

Records a short `.mcap` bag from a running sensor-suite container and copies it into
`bags/`. This one is *not* Python and does not use the venv — it needs Docker, and
SSH access unless you pass `--local`.

```bash
./capture_bag.sh --local --name vss_00000045                         # 3s by default
./capture_bag.sh dockware@beyonce --name vss_00000045 --duration 10
```

`--name` is the device serial the bag is saved as, written out in full as `vss_` plus
8 digits (`--name vss_00000045` → `bags/vss_00000045_calib_<timestamp>`), so the GUI picks up the serial from the
bag name and exports into `devices/vss_00000045/`. `--serial` picks the container to
record from and defaults to `calib` (the `vss_calib` calibration container); without
`--name` the bag is named after the container.

## 5. Troubleshooting

**`ModuleNotFoundError: No module named 'pupil_apriltags'`** — the venv isn't
active, or the install predates these deps being declared. Re-run
`python -m pip install -e .` with the venv active.

**`Could not load the Qt platform plugin "xcb"`** — install the system libraries in
step 1. To see which specific library is missing:
`QT_DEBUG_PLUGINS=1 ros2_calib`.

**`Illegal instruction (core dumped)` on import** — Open3D needs AVX. The package
imports Open3D lazily so the LiDAR-to-Camera workflow still works on older CPUs, but
LiDAR-to-LiDAR will not.

**Wrong Python picked up** — confirm with `which python`; it should print a path
inside `.venv/bin`.
