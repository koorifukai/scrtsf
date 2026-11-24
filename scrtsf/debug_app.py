"""Separate GUI for optical simulations and camera diagnostics."""

import tempfile
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np

from . import diagnostics, simulation, visual
from .app import Camera, add_file_queue, create_root, main as transfer_setup
from .display import OpticalWindow
from .session import Receiver, Sender


def run_debug(paths, level, screen, footprint=(320, 240), handheld=True, reflected=False, quarter_turns=0,
              max_profile=None):
    rng = np.random.default_rng()
    attempts = [0, 0]
    successes = [0, 0]
    with tempfile.TemporaryDirectory(prefix="scrtsf-debug-") as directory:
        sender = Sender(paths, time.monotonic(), max_profile=max_profile)
        receiver = Receiver(directory)
        window = "ScreenTransfer debug loopback"
        preview = "Simulated camera"
        try:
            optical = OpticalWindow(window, screen)
            cv2.namedWindow(preview, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(preview, 640, 480)
            while True:
                now = time.monotonic()
                profile, data = sender.frame(now)
                label = (f"DEBUG {level} | data {successes[0]}/{attempts[0]} "
                         f"ACK {successes[1]}/{attempts[1]} | {sender.label()}")
                optical.show(profile, data, label)
                display = visual.render(data, profile, optical.size, label)
                camera_image = simulation.distort(display, level, rng, footprint, handheld, reflected, quarter_turns)
                attempts[0] += 1
                try:
                    decoded = visual.decode(camera_image)
                except ValueError:
                    decoded = None
                if decoded is not None:
                    receiver.handle(decoded)
                    successes[0] += 1
                if receiver.state == "error":
                    raise RuntimeError(f"Simulated receiver: {receiver.message}")
                ack_profile, ack = receiver.frame()
                ack_image = simulation.distort(visual.render(ack, ack_profile, optical.size), level, rng,
                                               footprint, handheld, reflected, quarter_turns)
                attempts[1] += 1
                try:
                    decoded_ack = visual.decode(ack_image)
                except ValueError:
                    decoded_ack = None
                if decoded_ack is not None:
                    sender.handle(decoded_ack, time.monotonic())
                    successes[1] += 1
                cv2.imshow(preview, camera_image)
                if sender.done:
                    return (f"Debug complete: {len(receiver.saved)} file(s) verified in temporary storage.\n"
                            f"Data frames: {successes[0]}/{attempts[0]}; ACK frames: {successes[1]}/{attempts[1]}.")
                if cv2.waitKey(max(1, int((sender.frame_interval - (time.monotonic() - now)) * 1000))) & 0xFF == 27 or optical.closed():
                    return (f"Debug stopped. {len(receiver.saved)} file(s) verified.\n"
                            f"Data frames: {successes[0]}/{attempts[0]}; ACK frames: {successes[1]}/{attempts[1]}.")
        finally:
            cv2.destroyAllWindows()


def capture_test(index, resolution):
    camera = Camera(index, *resolution)
    started = time.monotonic()
    window = "Camera test - Space: analyze current frame, Esc: close"
    try:
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        while True:
            image, timestamp = camera.latest()
            now = time.monotonic()
            if (not timestamp and now - started > 8) or (timestamp and now - timestamp > 8):
                raise RuntimeError("Camera stopped delivering frames")
            if image is not None:
                cv2.imshow(window, image)
            key = cv2.waitKey(20) & 0xFF
            if key == 32 and image is not None:
                return image.copy()
            if key == 27:
                return None
    finally:
        camera.close()
        cv2.destroyAllWindows()


def main():
    root = create_root("ScreenTransfer debug")
    container = ttk.Frame(root, padding=16)
    container.pack(fill="both", expand=True)
    ttk.Label(container, text="ScreenTransfer debug", font=("Segoe UI", 16, "bold")).pack(anchor="w")
    ttk.Label(container, text="Local tests only. Temporary received files are deleted when a loopback ends.").pack(anchor="w", pady=8)
    screen = (root.winfo_screenwidth(), root.winfo_screenheight())
    noise = tk.StringVar(value="Medium")
    footprint = tk.StringVar(value="320 x 240")
    handheld = tk.BooleanVar(value=True)
    reflected = tk.BooleanVar(value=False)
    rotation = tk.StringVar(value="0")
    density = tk.StringVar(value="Auto (all profiles)")
    options = ttk.Frame(container)
    options.pack(fill="x", pady=8)
    ttk.Label(options, text="Distortion").grid(row=0, column=0, sticky="w")
    ttk.Combobox(options, textvariable=noise, values=tuple(simulation.LEVELS),
                 state="readonly", width=12).grid(row=0, column=1, padx=8)
    ttk.Label(options, text="Screen footprint (not webcam size)").grid(row=1, column=0, sticky="w")
    ttk.Combobox(options, textvariable=footprint, values=("320 x 240", "480 x 360", "640 x 480", "1280 x 720"),
                 state="readonly", width=12).grid(row=1, column=1, padx=8)
    ttk.Checkbutton(options, text="Handheld jitter and motion blur", variable=handheld).grid(row=2, column=0, sticky="w")
    ttk.Checkbutton(options, text="Simulated reflection", variable=reflected).grid(row=3, column=0, sticky="w")
    ttk.Label(options, text="Simulated rotation").grid(row=4, column=0, sticky="w")
    ttk.Combobox(options, textvariable=rotation, values=("0", "90", "180", "270"),
                 state="readonly", width=12).grid(row=4, column=1, padx=8)
    ttk.Label(options, text="Debug density ceiling").grid(row=5, column=0, sticky="w")
    ttk.Combobox(options, textvariable=density,
                 values=("Auto (all profiles)", *(visual.PROFILES[i].label for i in visual.PROFILE_ORDER)),
                 state="readonly", width=34).grid(row=5, column=1, padx=8)
    paths = add_file_queue(container)

    def loopback():
        try:
            dims = tuple(int(value.strip()) for value in footprint.get().split("x"))
            root.withdraw()
            root.update()
            result = run_debug(paths, noise.get(), screen, dims, handheld.get(), reflected.get(),
                               int(rotation.get()) // 90, selected_profile())
        except (OSError, ValueError, RuntimeError, cv2.error) as exc:
            root.deiconify()
            messagebox.showerror("Debug loopback", str(exc), parent=root)
        else:
            root.deiconify()
            messagebox.showinfo("Debug loopback", result, parent=root)

    ttk.Button(container, text="Start debug loopback", command=loopback).pack(anchor="e", pady=8)

    def show_report(image):
        messagebox.showinfo("Photo diagnostics", diagnostics.inspect_image(image), parent=root)

    def analyze_photo():
        path = filedialog.askopenfilename(parent=root, title="Select a camera photo",
                                          filetypes=(("Images", "*.jpg *.jpeg *.png *.bmp"), ("All files", "*.*")))
        if not path:
            return
        try:
            show_report(diagnostics.load_image(path))
        except (OSError, ValueError, cv2.error) as exc:
            messagebox.showerror("Photo diagnostics", str(exc), parent=root)

    ttk.Button(container, text="Analyze camera photo...", command=analyze_photo).pack(anchor="w", pady=8)
    live = ttk.LabelFrame(container, text="Live calibration capture", padding=8)
    live.pack(fill="x")
    camera = tk.StringVar(value="0")
    resolution = tk.StringVar(value="1280 x 720")
    ttk.Label(live, text="Camera index").grid(row=0, column=0)
    ttk.Entry(live, textvariable=camera, width=5).grid(row=0, column=1, padx=8)
    ttk.Combobox(live, textvariable=resolution, values=("640 x 480", "1280 x 720", "1920 x 1080"),
                 state="readonly", width=12).grid(row=0, column=2)

    def camera_test():
        try:
            index = int(camera.get())
            if index < 0:
                raise ValueError("Camera index must be non-negative")
            dims = tuple(int(value.strip()) for value in resolution.get().split("x"))
            root.withdraw()
            root.update()
            image = capture_test(index, dims)
            root.deiconify()
            if image is not None:
                show_report(image)
        except (OSError, ValueError, RuntimeError, cv2.error) as exc:
            root.deiconify()
            messagebox.showerror("Camera test", str(exc), parent=root)

    ttk.Button(live, text="Test live camera (Space captures)", command=camera_test).grid(
        row=1, column=0, columnspan=3, sticky="w", pady=8)

    def selected_profile():
        return next((i for i, profile in enumerate(visual.PROFILES) if profile.label == density.get()), None)

    def overridden_transfer():
        root.withdraw()
        try:
            dims = tuple(int(value.strip()) for value in resolution.get().split("x"))
            transfer_setup(resolution_override=dims, profile_override=selected_profile())
        finally:
            root.deiconify()

    ttk.Button(container, text="Open transfer setup with debug overrides", command=overridden_transfer).pack(anchor="w", pady=8)
    root.mainloop()


if __name__ == "__main__":
    main()
