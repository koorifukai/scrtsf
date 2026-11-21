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

Double-click `run_gui.bat`, or run `python -m scrtsf`. No standalone executable
is provided.

## Use: acknowledged transfer

1. Place the PCs so each webcam can see the entirety of the opposite screen,
   including all four corner markers. Reduce glare and tilt, and move closer
   if the link does not start. Do not cover either display.
2. On the receiver, select **Receiver**, choose an existing save folder, select
   the camera index and requested camera resolution, and click **Start**.
3. On the sender, select **Sender**, add files in the desired order, select the
   camera settings, and click **Start**. Each file must be at most 3 MiB.
4. Both displays switch to optical frames. The PCs handshake and test three
   profiles automatically; calibration takes at least 15 seconds. The sender
   then transfers the queue and stops when the receiver acknowledges completion.
   The receiver remains on its completion screen; press **Esc** to close it.
   Esc also cancels an active session on either side.

The line along the bottom shows the actual camera frame dimensions and whether
that PC has recently decoded the opposite screen. If either side says
"looking for screen", adjust its camera's view of the other display. The camera
index and resolution can be changed by stopping and restarting the session.

Try a different camera index (0, 1, ...) if the wrong camera opens. The
resolution is a request to the camera, not a guarantee. A 1080p camera and
good focus may help with denser profiles; lower-resolution cameras may require
the PCs to be closer. If calibration repeatedly fails, reposition the screens
and cameras. Both webcams must remain available throughout the transfer.

## Use: no-ACK transfer

1. Select **No ACK** on both PCs. On the receiver, choose a save folder and
   camera; its webcam must see the sender's entire screen.
2. On the sender, select the files and a **Send profile**. Start with `mono`,
   the most robust profile. The sender does not need a webcam.
3. Start both sides. The sender repeatedly displays the entire queue until
   you press Esc; the receiver collects blocks across passes and saves each
   verified file once. Watch the receiver for progress and errors, and stop
   the sender only after the receiver shows all files saved.

No-ACK mode cannot automatically calibrate, report errors to the sender, or
confirm delivery. If progress stalls, try better positioning or restart both
sides with a less dense profile; restarting loses in-memory progress. A
receiver can join after the sender starts. Up to 64 uniquely named files and
32 MiB of compressed data are supported per queue; each file still has the
3 MiB uncompressed limit. Keep the receiver running through the entire queue:
it retains incomplete files from multiple passes in memory.

## Debug loopback

On one PC, select **Sender**, add files, and check **Debug loopback** (leave
No ACK unchecked). Choose Light, Medium or Strong simulated distortion and
start. No webcam or second PC is needed. The sender and an in-process receiver
exchange rendered frames through simulated perspective, blur, random noise and
JPEG compression in both directions. The fullscreen view shows decoded/attempted
frame counts; a separate window shows the simulated receiver camera image.
The receiver verifies files in temporary storage, which is deleted when the
debug run ends. Esc stops the run. The simulation does not establish real
camera performance.

The receiver refuses to overwrite an existing file and reports the error to the
sender. Files are saved only after their size and SHA-256 checksum are verified.
Incomplete transfers are held in memory and are lost when the application
closes. Transfers are not encrypted; anyone who can see the screen can
potentially capture the file. Only exchange files with a trusted peer.

## How it works

The control link is a robust monochrome frame in the opposite direction. Data
uses monochrome or eight-color cells, corner markers, perspective correction,
interleaved Reed-Solomon error correction, and packet CRCs. Calibration selects
the profile with the best measured usable block rate; it can fall back to a
less dense profile if progress stalls. Camera frame rate and resolution are not
automatically tuned.

Files are compressed with zlib only when this makes them smaller. The receiver
reports missing blocks in windows of 32; the sender retransmits them until
acknowledged. No-ACK mode instead cycles through all blocks and repeats
metadata, so lost frames can be recovered in a later pass. Neither mode uses
fountain coding.
The frame interval and calibration thresholds are initial conservative values,
not measured throughput guarantees.

## Tests

```powershell
python -m unittest discover -s tests -v
```

The tests cover simulated perspective, blur and JPEG compression, and dropped
data and feedback frames. They do not substitute for testing two physical PCs.
