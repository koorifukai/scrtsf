"""Tkinter setup and windowed OpenCV sessions."""

import ctypes
import sys
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2

from . import visual
from .calibration import FRAME_SECONDS, PATTERN_DURATION, PatternReceiver, PatternSender
from .display import OpticalWindow
from .session import NoAckSender, Receiver, Sender


def configure_camera(capture, resolution=None, clock=time.monotonic):
    if resolution is not None:
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
        return
    capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    choices = []
    for width, height in ((640, 480), (1280, 720), (1920, 1080)):
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        ok, _ = capture.read()
        if not ok:
            continue
        started = clock()
        frames = []
        for _ in range(4):
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(frame)
        if len(frames) != 4:
            continue
        rate = len(frames) / max(0.001, clock() - started)
        actual_height, actual_width = frames[-1].shape[:2]
        choices.append((rate >= 12, actual_width * actual_height, min(rate, 60), width, height))
    if not choices:
        raise RuntimeError("No camera mode delivered test frames. Check camera permissions and availability.")
    _, _, _, width, height = max(choices)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)


class Camera:
    def __init__(self, index, width=None, height=None):
        backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
        self.capture = cv2.VideoCapture(index, backend)
        if not self.capture.isOpened():
            self.capture.release()
            raise RuntimeError(f"Could not open camera {index}. Check the index and Windows camera permissions.")
        try:
            configure_camera(self.capture, (width, height) if width is not None and height is not None else None)
        except (RuntimeError, cv2.error):
            self.capture.release()
            raise
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
    camera = None if isinstance(machine, NoAckSender) else Camera(camera_index, *(resolution or ()))
    window = "ScreenTransfer"
    started = time.monotonic()
    last_seen = 0.0
    last_decoded = 0.0
    next_frame = 0.0
    display = None
    decode_issue = "Find all 4 anchors"
    try:
        optical = OpticalWindow(window, screen)
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
                except ValueError as exc:
                    decoded = None
                    decode_issue = {
                        "Frame could not be decoded": "Unclear image: steady camera",
                        "Insufficient contrast": "Low contrast: reduce glare",
                    }.get(str(exc), str(exc))
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
                    link = "OK" if now - last_decoded < 3 and last_decoded else decode_issue
                    if isinstance(machine, Sender) and now - machine.last_ack >= 8:
                        link = "ACK lost: check peer screen"
                    label = f"{machine.label()} | Cam {camera_size} {link}"
                else:
                    label = machine.label()
                display = (profile, data, label)
                resized = optical.show(*display)
                if resized and isinstance(machine, Sender) and machine.state == "data":
                    machine.start_calibration(resuming=True)
                next_frame = time.monotonic() + machine.frame_interval
            if cv2.waitKey(1) & 0xFF == 27 or optical.closed():
                if isinstance(machine, Receiver):
                    return f"Stopped. Files saved: {len(machine.saved)}."
                if isinstance(machine, NoAckSender):
                    return "Stopped. Delivery cannot be confirmed in no-ACK mode."
                return "Stopped before the receiver confirmed completion."
    finally:
        if camera:
            camera.close()
        cv2.destroyAllWindows()


def show_calibration(screen):
    sender = PatternSender()
    try:
        optical = OpticalWindow("ScreenTransfer - Show calib imgs", screen)
        next_frame = 0.0
        while True:
            if time.monotonic() >= next_frame:
                slot = sender.slot
                profile, data = sender.frame()
                label = f"Calibration {slot + 1}/{len(sender.frames)} | {visual.PROFILES[profile].label} | Esc: stop"
                optical.show(profile, data, label)
                next_frame = time.monotonic() + FRAME_SECONDS
            if cv2.waitKey(10) & 0xFF == 27 or optical.closed():
                return
    finally:
        cv2.destroyAllWindows()


def take_calibration(camera_index, resolution):
    camera = Camera(camera_index, *(resolution or ()))
    receiver = PatternReceiver()
    window = "ScreenTransfer - Take calib"
    started = time.monotonic()
    deadline = None
    last_seen = None
    try:
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window, 800, 600)
        while True:
            now = time.monotonic()
            frame, timestamp = camera.latest()
            if now - (timestamp or started) > 8:
                raise RuntimeError("The camera is not delivering frames.")
            if frame is not None and timestamp != last_seen:
                last_seen = timestamp
                if deadline is None:
                    deadline = now + PATTERN_DURATION
                try:
                    decoded = visual.decode(frame)
                except visual.AnchorsNotFound:
                    pass
                except ValueError:
                    receiver.saw_anchors = True
                else:
                    receiver.handle(decoded)
                result = receiver.result(finished=time.monotonic() >= deadline)
                if result is not None:
                    return result
                preview = frame.copy()
                status = f"Scanning calibration | {max(1, int(deadline - now + 1))}s left | Esc: stop"
                cv2.putText(preview, status, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (255, 255, 255), 2, cv2.LINE_AA)
                cv2.imshow(window, preview)
            if cv2.waitKey(10) & 0xFF == 27 or (last_seen is not None and cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1):
                return None
            if deadline is not None and now >= deadline:
                return receiver.result(finished=True)
    finally:
        camera.close()
        cv2.destroyAllWindows()


def create_root(title):
    if sys.platform == "win32":
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except (AttributeError, OSError):
            pass
    root = tk.Tk()
    root.title(title)
    return root


def add_file_queue(container):
    paths = []
    files_box = ttk.LabelFrame(container, text="Files in order", padding=8)
    files_box.pack(fill="both", expand=True)
    files_list = tk.Listbox(files_box, height=5, selectmode="extended")
    files_list.pack(side="left", fill="both", expand=True)
    file_buttons = ttk.Frame(files_box)
    file_buttons.pack(side="left", padx=(8, 0), anchor="n")

    def add_files():
        for selected in filedialog.askopenfilenames(parent=container.winfo_toplevel(), title="Select files to send"):
            paths.append(selected)
            files_list.insert("end", selected)

    def remove_files():
        for index in reversed(files_list.curselection()):
            paths.pop(index)
            files_list.delete(index)

    ttk.Button(file_buttons, text="Add files", command=add_files).pack(fill="x")
    ttk.Button(file_buttons, text="Remove", command=remove_files).pack(fill="x", pady=4)
    return paths


def main(resolution_override=None, profile_override=None):
    root = create_root("ScreenTransfer setup" if resolution_override is None and profile_override is None
                       else "ScreenTransfer debug transfer setup")
    root.minsize(570, 430)
    screen = (root.winfo_screenwidth(), root.winfo_screenheight())
    role = tk.StringVar(value="send")
    camera = tk.StringVar(value="0")
    directory = tk.StringVar()
    no_ack = tk.BooleanVar(value=False)
    no_ack_profile = tk.StringVar(value=visual.PROFILES[
        profile_override if profile_override is not None else visual.CONTROL_PROFILE].label)

    container = ttk.Frame(root, padding=16)
    container.pack(fill="both", expand=True)
    ttk.Label(container, text="ScreenTransfer", font=("Segoe UI", 16, "bold")).pack(anchor="w")
    ttk.Label(container, text="Set up both PCs. Rotation and mirroring are corrected automatically.").pack(anchor="w", pady=(2, 10))
    roles = ttk.Frame(container)
    roles.pack(fill="x")
    ttk.Radiobutton(roles, text="Sender", variable=role, value="send").pack(side="left")
    ttk.Radiobutton(roles, text="Receiver", variable=role, value="receive").pack(side="left", padx=12)

    options = ttk.Frame(container)
    options.pack(fill="x", pady=10)
    ttk.Label(options, text="Camera index").grid(row=0, column=0, sticky="w")
    ttk.Entry(options, textvariable=camera, width=6).grid(row=0, column=1, padx=(8, 24))
    modes = ttk.Frame(container)
    modes.pack(fill="x", pady=(0, 8))
    ttk.Checkbutton(modes, text="No ACK (select on both PCs)", variable=no_ack).grid(row=0, column=0, sticky="w")
    profile_label = ttk.Label(modes, text="Sender profile")
    profile_label.grid(row=1, column=0, sticky="w", pady=(6, 0))
    profile_box = ttk.Combobox(modes, textvariable=no_ack_profile,
                              values=tuple(visual.PROFILES[i].label for i in visual.PROFILE_ORDER),
                              state="readonly", width=34)
    profile_box.grid(row=1, column=1, padx=8, pady=(6, 0))

    def update_profile_visibility(*_):
        if no_ack.get() and role.get() == "send":
            profile_label.grid()
            profile_box.grid()
        else:
            profile_label.grid_remove()
            profile_box.grid_remove()

    no_ack.trace_add("write", update_profile_visibility)
    role.trace_add("write", update_profile_visibility)
    update_profile_visibility()
    ttk.Label(container, text="ACK: automatic density/timing. No ACK: start with Mono coarse. Dimensions are grid cells.").pack(anchor="w", pady=(0, 8))
    paths = add_file_queue(container)
    dest = ttk.LabelFrame(container, text="Receiver: save folder", padding=8)
    dest.pack(fill="x", pady=10)
    ttk.Entry(dest, textvariable=directory).pack(side="left", fill="x", expand=True)

    def choose_folder():
        selected = filedialog.askdirectory(parent=root, title="Select the save folder")
        if selected:
            directory.set(selected)

    ttk.Button(dest, text="Browse", command=choose_folder).pack(side="left", padx=(8, 0))
    ttk.Label(container, text="Maximum 3 MiB per file. Existing files are not overwritten. Esc stops a session.").pack(anchor="w")

    def selected_camera():
        index = int(camera.get())
        if index < 0:
            raise ValueError("Camera index must be non-negative")
        return index

    def start():
        try:
            if role.get() == "send":
                selected = next(i for i, profile in enumerate(visual.PROFILES) if profile.label == no_ack_profile.get())
                machine = (NoAckSender(paths, selected)
                           if no_ack.get() else Sender(paths, time.monotonic(), max_profile=profile_override))
            else:
                machine = Receiver(directory.get(), no_ack=no_ack.get())
            index = 0 if isinstance(machine, NoAckSender) else selected_camera()
            root.withdraw()
            root.update()
            result = run_link(machine, index, resolution_override, screen)
            root.deiconify()
            messagebox.showinfo("ScreenTransfer", result, parent=root)
        except (OSError, ValueError, RuntimeError, cv2.error) as exc:
            root.deiconify()
            messagebox.showerror("ScreenTransfer", str(exc), parent=root)

    def calibrate():
        sending = role.get() == "send"
        result = None
        try:
            index = 0 if sending else selected_camera()
            root.withdraw()
            root.update()
            if sending:
                show_calibration(screen)
            else:
                result = take_calibration(index, resolution_override)
        except (ValueError, RuntimeError, cv2.error) as exc:
            root.deiconify()
            messagebox.showerror("ScreenTransfer calibration", str(exc), parent=root)
        else:
            root.deiconify()
            if result is not None:
                messagebox.showinfo("ScreenTransfer calibration", result, parent=root)

    calibration_button = ttk.Button(container, command=calibrate)
    calibration_button.pack(anchor="w", pady=(12, 0))

    def label_calibration_button(*_):
        calibration_button.configure(text="Show calib imgs" if role.get() == "send" else "Take calib")

    role.trace_add("write", label_calibration_button)
    label_calibration_button()
    ttk.Button(container, text="Start", command=start).pack(anchor="e", pady=(12, 0))
    root.mainloop()
