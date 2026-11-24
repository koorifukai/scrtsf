# ScreenTransfer

Transfer a queue of files between two Windows PCs using a screen and webcam.
Both PCs run the same program. No network connection is used. This is an experimental prototype:
actual speed and reliability depend heavily on the screens, cameras, lighting,
focus and positioning.

## Install

Install Python 3.10 or newer on both PCs and make sure `python` works in
PowerShell. Copy this folder to each PC, then run:

```powershell
python -m pip install -r requirements.txt
```

Double-click `run_gui.bat`, or run `python -m scrtsf`, for normal transfers.
Use `run_gui_debug.bat` or `python -m scrtsf.debug_app` for simulations, photo
analysis and live camera tests. No standalone executable is provided.
Update the application on both PCs together: calibration control packets,
coarse profiles and fragmented metadata require matching versions for transfers.

## Use: acknowledged transfer

1. Place the PCs so each webcam can see the entirety of the opposite screen,
   including all four corner markers. Reduce glare and tilt, and move closer
   if the link does not start. Do not cover either display.
2. On the receiver, select **Receiver**, choose an existing save folder, select
   the camera index, and click **Start**.
3. On the sender, select **Sender**, add files in the desired order, select the
   camera index, and click **Start**. Each file must be at most 3 MiB.
   A queue may contain at most 64 uniquely named files and 32 MiB of encoded data.
4. Both PCs open movable OpenCV windows. Place and resize them on the desired
   monitors so each camera can see the opposite window, including all anchors.
   The PCs handshake and test numbered images from coarse to fine at two frame
   durations. The receiver sends coarse reports; the sender waits for each final
   report, selects the best usable payload rate, and confirms the winner with
   another trial before transferring. Allow roughly two to three minutes for
   calibration under good conditions, longer if control frames are missed.
   The sender stops when the receiver acknowledges completion.
   The receiver remains on its completion screen; press **Esc** to close it.
   Esc also cancels an active session on either side.

The single status line below the encoded grid shows current-file and queue percentages,
block counts, approximate useful rate/ETA, and link status. Sender progress counts
confirmed blocks; receiver progress counts unique received blocks. Queue progress
is weighted by encoded blocks (with one unit for an empty file), not original
uncompressed bytes. **Verifying** is displayed before a nonempty file is saved;
100% reception is not a claim that verification succeeded.

The camera index remains explicit. Normal setup automatically probes common
camera resolutions and short-run capture rates, favoring detail without choosing
a very slow high-resolution mode. This is a quick heuristic, not exhaustive
camera tuning or a guaranteed capture rate. Camera settings and ACK density
ceilings are available through the debug GUI; no-ACK sender profiles are
selectable in normal setup.
The anchors automatically correct screen rotation, horizontal or vertical
reflection, and perspective. There is no manual flipping setting. The detector
tries both handednesses, refines marker corners, and fits screen geometry using
all 16 marker corners. This does not recover a cropped-out anchor or severe
motion blur.

Try a different camera index (0, 1, ...) if the wrong camera opens. A 1080p camera and
good focus may help with denser profiles; lower-resolution cameras may require
the PCs to be closer. If calibration repeatedly fails, reposition the screens
and cameras. Both webcams must remain available throughout the transfer.

### Small screen images and handheld cameras

An HD webcam does not imply an HD view of the screen: only the pixels covered
by the screen count. Calibration begins with `Mono coarse`, using 64 x 32 cells.
Both profile dropdowns show grid dimensions in **cells**, not webcam or window
pixels. Each cell carries one bit in monochrome or three bits using eight colors.

| Profile | Grid (columns x rows) |
| --- | --- |
| Mono coarse | 64 x 32 |
| Mono medium | 88 x 48 |
| Mono fine | 112 x 64 |
| Color fine | 112 x 64 |
| Color dense | 156 x 88 |
| Mono extra fine | 192 x 128 |
| Color extra fine | 192 x 128 |
| Mono ultra fine | 256 x 160 |
| Color ultra fine | 256 x 160 |

The extra/ultra presets target a sharply captured screen footprint of roughly
VGA (640 x 480 camera pixels), even when the overall webcam frame is HD or 2K.
At that footprint, their cells span roughly 3.0 and 2.3 camera pixels,
respectively, before perspective or cropping losses. They are aggressive
options, not a promised minimum resolution. Larger windows and stable focus help.
Ideal synthetic VGA captures decode all four new presets at full payload.
With Light simulated blur/perspective/JPEG, the new monochrome presets decoded
three of three trials each, but the color presets decoded zero of three.
Color needs cleaner reception; calibration can reject it. Real-camera
performance remains unverified.

The handshake, acknowledgements and metadata use large monochrome cells even
when data uses a denser profile. Calibration tests 0.7- and 0.35-second data holds;
control frames always use conservative 0.7-second holds. Actual updates may be
slower because of processing. Missed or checksum-invalid captures are discarded and retried rather than
combined across camera frames.

### Standalone, one-way calibration

1. On the sender, choose **Sender** and click **Show calib imgs**. The window
   loops through test images without waiting for feedback. No file or sender
   webcam is needed.
2. On the receiver, choose **Receiver**, select the camera, and click
   **Take calib**. Keep the whole calibration image in view. The camera scans
   for about 24 seconds unless the highest-capacity preset passes sooner.
3. Use the recommended preset in the sender's **No ACK** setup. The receiver
   identifies the preset automatically; there is no receiver setting to match.
   The test works regardless of whether the **No ACK** checkbox is selected.

The test goes from most to fewest usable file bytes per image at the same
0.7-second frame hold. Every image encodes the same short test message plus
random bytes to fill the preset's packet capacity. The receiver verifies the
whole packet and requires two of three distinct test frames at a preset.
Repeated camera captures of one frame do not count twice. If no pattern is
seen, the result is **Not detected**; if anchors are visible but no test frame
is readable, the result says so. **Start** in ACK mode still runs its own
two-way calibration.

For more detailed positioning tests, use **Test live camera** in the debug GUI.
Space captures a frame for analysis; Esc cancels. Aim for roughly 3.5 or more
camera pixels per cell in both directions, with margin for movement. Take
several captures while holding the camera as you intend to during transfer.

The simulator tests a screen within a 320 x 240 bounding box by default, not
a guaranteed minimum for real cameras. Brace the camera if possible, use good
lighting to reduce exposure time, and keep all anchors in view. Slower display
updates give more capture opportunities but do not fix blur within an exposure.
The coarse profile is deliberately slow: an incompressible 3 MiB file can take
hours, even before retries. Start with a small file, then increase density only
after testing. Transfers automatically recalibrate without discarding blocks
after a resize, stalled progress with a live ACK link, or sustained throughput
deterioration. Moving between monitors can also change the optical conditions.
Keep the whole window visible and avoid resizing during calibration; use
calibration again if you substantially change positioning.

## Use: no-ACK transfer

1. Select **No ACK** on both PCs. On the receiver, choose a save folder and
   camera; its webcam must see the sender's entire screen.
2. On the sender, select the files and choose **Sender profile**, which appears
   only for a no-ACK sender. Start with `Mono coarse`; denser profiles can be faster
   but require a clearer, larger camera view. Frames use 0.7-second holds, and
   the sender does not need a webcam.
3. Start both sides. The sender repeatedly displays the entire queue until
   you press Esc; the receiver collects blocks across passes and saves each
   verified file once. Watch the receiver for progress and errors, and stop
   the sender only after the receiver shows all files saved.

No-ACK mode cannot automatically calibrate, report errors to the sender, or
confirm delivery. Its sender percentage describes the transmission cycle,
not delivery. If progress stalls, try better positioning or restart with a
coarser sender profile; restarting loses in-memory progress. Webcam capture
resolution remains automatic unless overridden through debug setup. A
receiver can join after the sender starts. Up to 64 uniquely named files and
32 MiB of compressed data are supported per queue; each file still has the
3 MiB uncompressed limit. Keep the receiver running through the entire queue:
it retains incomplete files from multiple passes in memory.

## Debug GUI and loopback

Open `run_gui_debug.bat`. Add files and choose Light, Medium or Strong distortion,
the simulated screen footprint, handheld motion, reflection and rotation, then
click **Start debug loopback**. No webcam or second PC is needed for loopback.
The normal transfer GUI contains no debug controls. A debug density ceiling can
limit trials. **Open transfer setup with debug overrides** uses the debug GUI's
requested camera resolution and density ceiling for a real transfer. The density
is a maximum in ACK mode and a fixed profile in no-ACK mode.
The sender and an in-process receiver exchange rendered frames through simulated
downsampling, perspective, blur, random noise and JPEG compression in both
directions. Handheld mode adds changing corner jitter and mild directional motion
blur. The movable sender view shows decoded/attempted
frame counts; a separate window shows the simulated receiver camera image.
The receiver verifies files in temporary storage, which is deleted when the
debug run ends. Esc stops the run. The simulation does not establish real
camera performance.

### Analyze a camera photo

Click **Analyze camera photo...** in the debug GUI and choose an image from
the camera. Orientation correction is automatic. The report shows marker IDs,
profile, approximate camera
pixels per cell, data-cell brightness, expected padding and whether the packet
passed its checksum. Photos stay on your PC. A dark strip at the end of the
grid may be unused padding, not damaged data. A single still image cannot
establish transfer performance or diagnose all camera and screen effects.

The receiver refuses to overwrite an existing file and reports the error to the
sender. Files are saved only after their size and SHA-256 checksum are verified.
Incomplete transfers are held in memory and are lost when the application
closes. Transfers are not encrypted; anyone who can see the screen can
potentially capture the file. Only exchange files with a trusted peer.

## How it works

The control link is a robust monochrome frame in the opposite direction. Data
uses monochrome or eight-color cells, corner markers, perspective correction,
interleaved Reed-Solomon error correction, and packet CRCs. Calibration counts
distinct valid test images and measures useful block bytes per elapsed second,
not repeated captures of an image. A minimum success fraction and failure-streak
limit reject unreliable trials; the best remaining setting is independently
confirmed. Coarse ready/report/finish handshakes prevent lost dense images or
delayed ACKs from prematurely ending a trial. The ACK screen does not consume
sender-screen bandwidth, but camera and processing latency still matter.

The optical grid has a fixed reserved status area, so calibration and file
transfer use the same display geometry. If normal monochrome sampling fails,
the decoder tries a bounded set of smaller/shifted sampling windows and thresholds; the packet must
still pass its CRC.

Files are compressed with zlib only when this makes them smaller. The receiver
reports missing 128-byte blocks in windows of 64; the sender retransmits them until
acknowledged. Metadata is fragmented so long names also fit coarse frames.
The larger block window lets fast data frames carry different blocks while
waiting for a coarse ACK update.
No-ACK mode instead cycles through all blocks and repeats
metadata, rotating fragment order and varying repetitions to avoid locking
onto periodic missed captures, so lost frames can be recovered in a later pass. Neither mode uses
fountain coding.
The frame interval and calibration thresholds are initial conservative values,
not measured throughput guarantees.

## Tests

```powershell
python -m unittest discover -s tests -v
```

The tests cover all rotations/reflections, simulated small screen footprints,
handheld motion, blur, JPEG compression, fragmented metadata, and dropped data
and feedback frames, delayed final reports, confirmation, queue progress and
resized displays. They do not substitute for testing two physical PCs.
