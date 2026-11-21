"""Tkinter setup and the fullscreen OpenCV session."""

import ctypes
import sys
import tempfile
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np

from . import simulation, visual
from .session import FRAME_SECONDS, NoAckSender, Receiver, Sender


class Camera:
    def __init__(self, index, width, height):
        backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
        self.capture = cv2.VideoCapture(index, backend)
        if not self.capture.isOpened():
            self.capture.release()
            raise RuntimeError(f"Could not open camera {index}. Check the index and Windows camera permissions.")
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.lock = threading.Lock()
        self.frame = None
        self.timestamp = 0.0
        self.stopped = threading.Event()
        self.thread = threading.Thread(target=self._read, daemon=True)
        self.thread.start()

    def _read(self):
        while not self.stopped.is_set():
            ok, frame = self.capture.read()
            if not ok:
                self.stopped.wait(0.1)
                continue
            with self.lock:
                self.frame = frame
                self.timestamp = time.monotonic()

    def latest(self):
        with self.lock:
            return self.frame, self.timestamp

    def close(self):
        self.stopped.set()
        self.thread.join(timeout=2)
        self.capture.release()


def run_link(machine, camera_index, resolution, screen):
    camera = None if isinstance(machine, NoAckSender) else Camera(camera_index, *resolution)
    window = "ScreenTransfer"
    started = time.monotonic()
    last_seen = 0.0
    last_decoded = 0.0
    next_frame = 0.0
    display = None
    try:
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(window, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        while True:
            now = time.monotonic()
            frame, timestamp = camera.latest() if camera else (None, 0)
            if camera:
                if timestamp and now - timestamp > 8:
                    raise RuntimeError("The camera stopped delivering frames.")
                if not timestamp and now - started > 8:
                    raise RuntimeError("The camera did not deliver any frames.")
            if frame is not None and timestamp != last_seen:
                last_seen = timestamp
                try:
                    decoded = visual.decode(frame)
                except ValueError:
                    decoded = None
                if decoded is not None:
                    last_decoded = now
                    try:
                        if isinstance(machine, Sender):
                            machine.handle(decoded, now)
                        else:
                            machine.handle(decoded)
                    except (ValueError, UnicodeError) as exc:
                        machine.message = f"Invalid optical packet: {exc}"
            if isinstance(machine, Sender) and machine.done:
                return "All files were acknowledged by the receiver."
            if isinstance(machine, Receiver) and machine.no_ack and machine.state == "error":
                raise RuntimeError(f"Receiver: {machine.message}")
            if now >= next_frame:
                if isinstance(machine, (Sender, NoAckSender)):
                    profile, data = machine.frame(now)
                else:
                    profile, data = machine.frame()
                if camera:
                    camera_size = f"{frame.shape[1]}x{frame.shape[0]}" if frame is not None else "starting"
                    link = "screen decoded" if now - last_decoded < 3 and last_decoded else "looking for screen"
                    label = f"{machine.label()} | camera {camera_size}: {link}"
                else:
                    label = machine.label()
                display = visual.render(data, profile, screen, label)
                next_frame = now + FRAME_SECONDS
            if display is not None:
                cv2.imshow(window, display)
            if cv2.waitKey(1) & 0xFF == 27:
                if isinstance(machine, Receiver):
                    return f"Stopped. Files saved: {len(machine.saved)}."
                if isinstance(machine, NoAckSender):
                    return "Stopped. Delivery cannot be confirmed in no-ACK mode."
                return "Stopped before the receiver confirmed completion."
    finally:
        if camera:
            camera.close()
        cv2.destroyAllWindows()


def run_debug(paths, level, screen):
    rng = np.random.default_rng()
    attempts = [0, 0]
    successes = [0, 0]
    with tempfile.TemporaryDirectory(prefix="scrtsf-debug-") as directory:
        sender = Sender(paths, time.monotonic())
        receiver = Receiver(directory)
        window = "ScreenTransfer debug loopback"
        preview = "Simulated camera"
        try:
            cv2.namedWindow(window, cv2.WINDOW_NORMAL)
            cv2.setWindowProperty(window, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            cv2.namedWindow(preview, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(preview, 640, 360)
            while True:
                now = time.monotonic()
                profile, data = sender.frame(now)
                label = (f"DEBUG {level} | data {successes[0]}/{attempts[0]} "
                         f"ACK {successes[1]}/{attempts[1]} | {sender.label()}")
                display = visual.render(data, profile, screen, label)
                camera_image = simulation.distort(display, level, rng)
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
                ack_image = simulation.distort(visual.render(ack, ack_profile, screen), level, rng)
                attempts[1] += 1
                try:
                    decoded_ack = visual.decode(ack_image)
                except ValueError:
                    decoded_ack = None
                if decoded_ack is not None:
                    sender.handle(decoded_ack, time.monotonic())
                    successes[1] += 1
                cv2.imshow(window, display)
                cv2.imshow(preview, camera_image)
                if sender.done:
                    return (f"Debug complete: {len(receiver.saved)} file(s) verified in temporary storage.\n"
                            f"Data frames: {successes[0]}/{attempts[0]}; ACK frames: {successes[1]}/{attempts[1]}.")
                if cv2.waitKey(max(1, int((FRAME_SECONDS - (time.monotonic() - now)) * 1000))) & 0xFF == 27:
                    return (f"Debug stopped. {len(receiver.saved)} file(s) verified.\n"
                            f"Data frames: {successes[0]}/{attempts[0]}; ACK frames: {successes[1]}/{attempts[1]}.")
        finally:
            cv2.destroyAllWindows()


def main():
    if sys.platform == "win32":
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except (AttributeError, OSError):
            pass
    root = tk.Tk()
    root.title("ScreenTransfer setup")
    root.minsize(570, 430)
    screen = (root.winfo_screenwidth(), root.winfo_screenheight())
    role = tk.StringVar(value="send")
    camera = tk.StringVar(value="0")
    resolution = tk.StringVar(value="1280 x 720")
    directory = tk.StringVar()
    no_ack = tk.BooleanVar(value=False)
    debug = tk.BooleanVar(value=False)
    profile = tk.StringVar(value=visual.PROFILES[0].name)
    noise = tk.StringVar(value="Medium")
    paths = []

    container = ttk.Frame(root, padding=16)
    container.pack(fill="both", expand=True)
    ttk.Label(container, text="ScreenTransfer", font=("Segoe UI", 16, "bold")).pack(anchor="w")
    ttk.Label(container, text="Set up both PCs, or run a local simulated loopback.").pack(anchor="w", pady=(2, 10))
    roles = ttk.Frame(container)
    roles.pack(fill="x")
    ttk.Radiobutton(roles, text="Sender", variable=role, value="send").pack(side="left")
    ttk.Radiobutton(roles, text="Receiver", variable=role, value="receive").pack(side="left", padx=12)

    options = ttk.Frame(container)
    options.pack(fill="x", pady=10)
    ttk.Label(options, text="Camera index").grid(row=0, column=0, sticky="w")
    ttk.Entry(options, textvariable=camera, width=6).grid(row=0, column=1, padx=(8, 24))
    ttk.Label(options, text="Requested resolution").grid(row=0, column=2)
    ttk.Combobox(options, textvariable=resolution, values=("1280 x 720", "1920 x 1080", "640 x 480"),
                 state="readonly", width=16).grid(row=0, column=3, padx=8)
    modes = ttk.Frame(container)
    modes.pack(fill="x", pady=(0, 8))
    ttk.Checkbutton(modes, text="No ACK (select on both PCs)", variable=no_ack).grid(row=0, column=0, sticky="w")
    ttk.Label(modes, text="Send profile").grid(row=0, column=1, padx=(10, 4))
    ttk.Combobox(modes, textvariable=profile, values=tuple(p.name for p in visual.PROFILES),
                 state="readonly", width=12).grid(row=0, column=2)
    ttk.Checkbutton(modes, text="Debug loopback (sender)", variable=debug).grid(row=1, column=0, sticky="w")
    ttk.Label(modes, text="Distortion").grid(row=1, column=1, padx=(10, 4))
    ttk.Combobox(modes, textvariable=noise, values=tuple(simulation.LEVELS),
                 state="readonly", width=12).grid(row=1, column=2)

    files_box = ttk.LabelFrame(container, text="Sender: files in order", padding=8)
    files_box.pack(fill="both", expand=True)
    files_list = tk.Listbox(files_box, height=5, selectmode="extended")
    files_list.pack(side="left", fill="both", expand=True)
    file_buttons = ttk.Frame(files_box)
    file_buttons.pack(side="left", padx=(8, 0), anchor="n")

    def add_files():
        for selected in filedialog.askopenfilenames(parent=root, title="Select files to send"):
            paths.append(selected)
            files_list.insert("end", selected)

    def remove_files():
        for index in reversed(files_list.curselection()):
            paths.pop(index)
            files_list.delete(index)

    ttk.Button(file_buttons, text="Add files", command=add_files).pack(fill="x")
    ttk.Button(file_buttons, text="Remove", command=remove_files).pack(fill="x", pady=4)
    dest = ttk.LabelFrame(container, text="Receiver: save folder", padding=8)
    dest.pack(fill="x", pady=10)
    ttk.Entry(dest, textvariable=directory).pack(side="left", fill="x", expand=True)

    def choose_folder():
        selected = filedialog.askdirectory(parent=root, title="Select the save folder")
        if selected:
            directory.set(selected)

    ttk.Button(dest, text="Browse", command=choose_folder).pack(side="left", padx=(8, 0))
    ttk.Label(container, text="Maximum 3 MiB per file. Existing files are not overwritten. Esc stops a session.").pack(anchor="w")

    def start():
        try:
            if debug.get() and (role.get() != "send" or no_ack.get()):
                raise ValueError("Debug loopback requires Sender and ACK mode.")
            machine = None
            index, dims = 0, (1280, 720)
            if not debug.get():
                if role.get() == "send":
                    profile_id = next(i for i, item in enumerate(visual.PROFILES) if item.name == profile.get())
                    machine = NoAckSender(paths, profile_id) if no_ack.get() else Sender(paths, time.monotonic())
                else:
                    machine = Receiver(directory.get(), no_ack=no_ack.get())
                if not isinstance(machine, NoAckSender):
                    index = int(camera.get())
                    if index < 0:
                        raise ValueError("Camera index must be non-negative")
                    dims = tuple(int(v.strip()) for v in resolution.get().split("x"))
                    if len(dims) != 2:
                        raise ValueError("Invalid camera resolution")
            root.withdraw()
            root.update()
            result = run_debug(paths, noise.get(), screen) if debug.get() else run_link(machine, index, dims, screen)
            root.deiconify()
            messagebox.showinfo("ScreenTransfer", result, parent=root)
        except (OSError, ValueError, RuntimeError, cv2.error) as exc:
            root.deiconify()
            messagebox.showerror("ScreenTransfer", str(exc), parent=root)

    ttk.Button(container, text="Start", command=start).pack(anchor="e", pady=(12, 0))
    root.mainloop()
