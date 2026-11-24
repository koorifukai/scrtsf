import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from scrtsf import app, debug_app, diagnostics
from scrtsf import protocol as p
from scrtsf import simulation, visual
from scrtsf.calibration import (CAL_CONTROL, CAL_PATTERN, PATTERN_DURATION, PATTERN_MESSAGE, PATTERN_ORDER,
                                Calibration, PatternReceiver, PatternSender, TrialReceiver)
from scrtsf.display import OpticalWindow
from scrtsf.session import MetadataBuffer, NO_ACK_META, NoAckSender, Receiver, Sender, metadata_fragment, parse_entries


class OpticalTests(unittest.TestCase):
    def test_finer_profiles_decode_at_vga_footprint(self):
        for profile_id in range(5, len(visual.PROFILES)):
            profile = visual.PROFILES[profile_id]
            rng = np.random.default_rng(profile_id)
            packet = p.packet(p.DATA, b"12345678", 1, rng.bytes(profile.capacity))
            screen = visual.render(packet, profile_id, (960, 720))
            frames = [cv2.resize(screen, (640, 480), interpolation=cv2.INTER_AREA)]
            if not profile.color:
                frames.append(simulation.distort(screen, "Light", rng, (640, 480)))
            for index, frame in enumerate(frames):
                with self.subTest(profile=profile.label, distortion=index):
                    self.assertEqual(visual.decode(frame), (profile_id, p.unpack(packet)))

    def test_status_text_does_not_change_calibration_grid_size(self):
        packet = p.packet(p.HELLO, b"12345678", 1)
        short = visual.render(packet, visual.CONTROL_PROFILE, (960, 720), "Calibrating")
        with patch("scrtsf.visual.cv2.putText", wraps=cv2.putText) as text:
            long = visual.render(packet, visual.CONTROL_PROFILE, (960, 720), "File 50%\nQueue 25%\nETA 10s\nCamera stable")
        self.assertEqual(text.call_count, 1)
        self.assertNotIn("\n", text.call_args.args[1])
        self.assertTrue(np.array_equal(short[:-36], long[:-36]))
        self.assertEqual(visual.decode(short), visual.decode(long))

    def test_windowed_display_resizes_without_covering_grid(self):
        packet = p.packet(p.DATA, b"12345678", 1, b"display test")
        label = "File 50.0% | queue 25.0%\n2/4 blocks confirmed\nETA ~10s | stable\nCamera: screen decoded"
        with patch("scrtsf.display.cv2.namedWindow") as named, \
             patch("scrtsf.display.cv2.resizeWindow"), \
             patch("scrtsf.display.cv2.getWindowImageRect", return_value=(100, 100, 800, 600)), \
             patch("scrtsf.display.cv2.imshow") as show, \
             patch("scrtsf.display.cv2.getWindowProperty", return_value=0):
            window = OpticalWindow("test", (1920, 1080))
            window.show(visual.CONTROL_PROFILE, packet, label)
            self.assertTrue(window.show(visual.CONTROL_PROFILE, packet, label))
            rendered = show.call_args.args[1]
            self.assertEqual(rendered.shape[:2], (600, 800))
            self.assertEqual(visual.decode(rendered), (visual.CONTROL_PROFILE, p.unpack(packet)))
            self.assertTrue(window.closed())
            named.assert_called_once_with("test", cv2.WINDOW_NORMAL)

    def test_all_profiles_decode_every_rotation_and_reflection(self):
        for profile, shape in enumerate(visual.PROFILES):
            packet = p.packet(p.DATA, b"12345678", 37, np.random.default_rng(profile).bytes(shape.capacity))
            image = visual.render(packet, profile, (1280, 720))
            for reflected in (False, True):
                for quarter_turns in range(4):
                    with self.subTest(profile=profile, reflected=reflected, rotation=quarter_turns):
                        frame = cv2.flip(image, 1) if reflected else image
                        frame = np.ascontiguousarray(np.rot90(frame, quarter_turns))
                        self.assertEqual(visual.decode(frame), (profile, p.unpack(packet)))

    def test_large_cells_at_small_footprint_with_handheld_motion(self):
        rng = np.random.default_rng(1)
        profile = visual.CONTROL_PROFILE
        packet = p.packet(p.DATA, b"12345678", 1, rng.bytes(visual.PROFILES[profile].capacity))
        screen = visual.render(packet, profile, (1280, 720))
        for tick in range(12):
            with self.subTest(tick=tick):
                frame = simulation.distort(screen, "Medium", rng, (320, 240), True, tick % 2 == 1, tick % 4)
                self.assertEqual(visual.decode(frame), (profile, p.unpack(packet)))

    def test_shifted_mono_cells_recover_with_fallback(self):
        profile = visual.PROFILES[0]
        payload = np.random.default_rng(42).bytes(450)
        packet = p.packet(p.DATA, b"12345678", 42, payload)
        image = visual.render(packet, 0, (profile.width, profile.height + 36))
        origin = visual.MARGIN + visual.MARKER + visual.GAP
        height, width = profile.rows * visual.CELL, profile.cols * visual.CELL
        grid = image[origin:origin + height, origin:origin + width].copy()
        grid = cv2.warpAffine(grid, np.float32([[1, 0, 4], [0, 1, 0]]),
                              (width, height), borderValue=(255, 255, 255))
        image[origin:origin + height, origin:origin + width] = cv2.GaussianBlur(grid, (5, 5), 0.6)
        _, _, warped = visual.locate(image)
        with self.assertRaises(ValueError):
            visual.decode_samples(visual.sample_grid(warped, profile), profile)
        self.assertEqual(visual.decode(image), (0, p.unpack(packet)))

    def test_photo_diagnostics_detect_mirrored_frame_and_padding(self):
        packet = p.packet(p.HELLO, b"12345678", 12)
        image = cv2.flip(visual.render(packet, 0, (1280, 720)), 1)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "\u753b\u50cf.png"
            ok, encoded = cv2.imencode(".png", image)
            self.assertTrue(ok)
            encoded.tofile(str(path))
            loaded = diagnostics.load_image(path)
        report = diagnostics.inspect_image(loaded)
        self.assertIn("Marker IDs detected: 0, 1, 2, 3", report)
        self.assertIn("Unused padding: 1048 of 7168", report)
        self.assertIn("decoded and CRC verified (hello", report)

    def test_interleaved_error_correction_recovers_damaged_cells(self):
        profile = visual.PROFILES[0]
        data = p.packet(p.DATA, b"12345678", 5, os.urandom(300))
        image = visual.render(data, 0, (profile.width, profile.height + 36))
        origin = visual.MARGIN + visual.MARKER + visual.GAP
        for byte_index in range(24):
            cell = byte_index * 8
            row, col = divmod(cell, profile.cols)
            y, x = origin + row * visual.CELL, origin + col * visual.CELL
            image[y:y + visual.CELL, x:x + visual.CELL] ^= 255
        self.assertEqual(visual.decode(image)[1], p.unpack(data))

    def test_profiles_survive_perspective_blur_and_jpeg(self):
        for profile in range(len(visual.PROFILES)):
            with self.subTest(profile=profile):
                payload = os.urandom(min(700, visual.PROFILES[profile].capacity))
                packet = p.packet(p.DATA, b"12345678", 17, payload)
                screen = visual.render(packet, profile, (1600, 900))
                src = np.float32([[0, 0], [1599, 0], [1599, 899], [0, 899]])
                dst = np.float32([[85, 32], [1500, 75], [1540, 850], [60, 820]])
                image = cv2.warpPerspective(screen, cv2.getPerspectiveTransform(src, dst), (1600, 900))
                image = cv2.GaussianBlur(image, (3, 3), 0.55)
                ok, jpeg = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 90])
                self.assertTrue(ok)
                decoded_profile, decoded = visual.decode(cv2.imdecode(jpeg, cv2.IMREAD_COLOR))
                self.assertEqual(decoded_profile, profile)
                self.assertEqual(decoded, (p.DATA, b"12345678", 17, payload))

    def test_packet_corruption_is_detected(self):
        data = bytearray(p.packet(p.DATA, b"12345678", 0, b"test"))
        data[-5] ^= 1
        with self.assertRaises(ValueError):
            p.unpack(data)


class PatternTests(unittest.TestCase):
    def test_loop_uses_same_message_in_full_size_frames_and_detects_best_profile(self):
        sender = PatternSender()
        receiver = PatternReceiver()
        frames = [sender.frame() for _ in range(len(sender.frames))]
        self.assertEqual(sender.frame(), frames[0])
        strongest, fallback = PATTERN_ORDER[0], PATTERN_ORDER[-1]
        fallback_frames = [(profile, data) for profile, data in frames if profile == fallback]
        for profile, data in fallback_frames[:2]:
            receiver.handle(visual.decode(visual.render(data, profile, (960, 720))))
        self.assertIsNone(receiver.result())
        self.assertIn(visual.PROFILES[fallback].label, receiver.result(finished=True))

        for profile, data in frames:
            kind, _, _, payload = p.unpack(data)
            self.assertEqual(kind, CAL_PATTERN)
            self.assertTrue(payload.startswith(PATTERN_MESSAGE))
            self.assertEqual(len(payload), visual.PROFILES[profile].capacity)
        profile, data = frames[0]
        self.assertEqual(profile, strongest)
        first = visual.decode(visual.render(data, profile, (1280, 720)))
        receiver.handle(first)
        receiver.handle(first)
        self.assertIsNone(receiver.result())
        profile, data = frames[1]
        receiver.handle(visual.decode(visual.render(data, profile, (1280, 720))))
        self.assertIn(visual.PROFILES[strongest].label, receiver.result())

    def test_capture_distinguishes_no_pattern_from_unreadable_payload(self):
        for issue, expected in ((visual.AnchorsNotFound(), "Not detected"),
                                (ValueError(), "Screen anchors detected, but no calibration frame decoded.")):
            with self.subTest(issue=type(issue)), \
                 patch.object(app, "Camera") as camera, \
                 patch.object(app.time, "monotonic", side_effect=[100, 100, 100, 100 + PATTERN_DURATION,
                                                                 100 + PATTERN_DURATION]), \
                 patch.object(app.cv2, "namedWindow"), patch.object(app.cv2, "resizeWindow"), \
                 patch.object(app.cv2, "imshow"), patch.object(app.cv2, "destroyAllWindows"), \
                 patch.object(app.cv2, "getWindowProperty", return_value=1), \
                 patch.object(app.cv2, "waitKey", return_value=-1), \
                 patch.object(app.visual, "decode", side_effect=issue):
                image = np.zeros((480, 640, 3), dtype=np.uint8)
                camera.return_value.latest.side_effect = [(image, 100), (image, 100 + PATTERN_DURATION)]
                self.assertEqual(app.take_calibration(1, (640, 480)), expected)
                camera.assert_called_once_with(1, 640, 480)
                camera.return_value.close.assert_called_once()


class SessionTests(unittest.TestCase):
    def test_no_ack_finer_profile_transfers_multiple_large_frames(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, dest = root / "file.bin", root / "dest"
            dest.mkdir()
            content = np.random.default_rng(28).bytes(p.BLOCK * 220 + 13)
            source.write_bytes(content)
            sender = NoAckSender([source], visual.PROFILE_ORDER[-1])
            receiver = Receiver(dest, no_ack=True)
            largest_payload = 0
            for tick in range(30):
                profile, data = sender.frame(tick * 0.7)
                if profile == sender.profile:
                    largest_payload = max(largest_payload, len(p.unpack(data)[3]))
                image = visual.render(data, profile, (960, 720))
                frame = cv2.resize(image, (640, 480), interpolation=cv2.INTER_AREA)
                receiver.handle(visual.decode(frame))
                receiver.frame()
                if receiver.saved:
                    break
            self.assertGreater(largest_payload, p.WINDOW * (p.BLOCK + 6))
            self.assertEqual((dest / source.name).read_bytes(), content)

    def test_sustained_throughput_drop_recalibrates_without_resetting_progress(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "file"
            path.write_bytes(np.random.default_rng(16).bytes(3000))
            sender = Sender([path], 0)
            sender.next_file(0)
            sender.state = "data"
            sender.data_interval = 0.35
            sender.expected_rate = 1024
            sender.received = 2
            sender.last_ack = sender.last_progress = 31
            sender.progress_samples.extend(((21, 1), (30, 2)))
            sender.frame(31)
            self.assertEqual(sender.state, "cal")
            self.assertTrue(sender.resuming)
            self.assertEqual(sender.received, 2)

    def test_empty_no_ack_queue_reports_cycle_not_delivery(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "empty"
            path.write_bytes(b"")
            sender = NoAckSender([path])
            sender.frame(0)
            sender.frame(0.7)
            self.assertIn("cycle 100.0%", sender.label())
            self.assertIn("Delivery unconfirmed", sender.label())
            self.assertNotIn("\n", sender.label())

    def test_fast_frames_pipeline_different_blocks_before_next_ack(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "file.bin"
            path.write_bytes(np.random.default_rng(12).bytes(p.BLOCK * 64))
            sender = Sender([path], 0)
            sender.next_file(0)
            sender.state = "data"
            sender.profile = 2
            sender.data_interval = 0.35
            sender.mask = (1 << 64) - 1
            first = parse_entries(p.unpack(sender.frame(0)[1])[3])
            second = parse_entries(p.unpack(sender.frame(0.35)[1])[3])
            self.assertEqual(len({index for index, _ in first + second}), 64)
            receiver = Receiver(temp)
            receiver.incoming = p.Incoming(sender.meta)
            receiver.state = "receiving"
            profile, packet = receiver.frame()
            self.assertEqual(visual.decode(visual.render(packet, profile, (960, 720))), (profile, p.unpack(packet)))

    def test_calibration_waits_for_final_report_and_measures_goodput(self):
        calibration = Calibration((visual.CONTROL_PROFILE, 4), 0.7)
        receiver = TrialReceiver()
        now = 0
        delayed = False
        for seq in range(200):
            profile, encoded = calibration.frame(b"12345678", seq, now)
            kind, _, _, payload = p.unpack(encoded)
            now += calibration.interval
            if kind == CAL_CONTROL:
                receiver.control(payload, seq)
            else:
                if not (profile == 4 and calibration.hold == 0.35):
                    receiver.add(profile, payload)
                    receiver.add(profile, payload)
                if calibration.phase == "report" and calibration.duration is None:
                    calibration.handle({"trial": calibration.trial, "phase": "report",
                                        "received": calibration.sent, "streak": 0})
                    self.assertEqual(calibration.phase, "report")
            if calibration.phase == "report" and receiver.phase == "report" and not delayed:
                trial = calibration.trial
                for repeat in range(3):
                    calibration.frame(b"12345678", seq + repeat, now)
                    now += calibration.interval
                self.assertEqual(calibration.trial, trial)
                self.assertEqual(calibration.phase, "report")
                delayed = True
            calibration.handle(receiver.status())
            if calibration.done:
                break
        self.assertTrue(delayed)
        self.assertTrue(calibration.done)
        self.assertEqual(calibration.selected, (4, 0.7))
        self.assertTrue(calibration.confirming)

    def test_recalibration_preserves_blocks(self):
        with tempfile.TemporaryDirectory() as temp, patch("scrtsf.session.CAL_SECONDS", 0.1):
            now = 0
            source = Path(temp) / "source.bin"
            content = np.random.default_rng(1).bytes(600)
            source.write_bytes(content)
            dest = Path(temp) / "dest"
            dest.mkdir()
            sender = Sender([source], now, max_profile=visual.CONTROL_PROFILE)
            receiver = Receiver(dest)
            resumed = False
            for _ in range(200):
                profile, data = sender.frame(now)
                receiver.handle((profile, p.unpack(data)))
                profile, ack = receiver.frame()
                sender.handle((profile, p.unpack(ack)), now)
                if not resumed and sender.received > 0:
                    incoming = receiver.incoming
                    count = incoming.received
                    sender.start_calibration(resuming=True)
                    resumed = True
                now += sender.frame_interval
                if sender.done:
                    break
            self.assertTrue(resumed)
            self.assertIs(receiver.incoming, incoming)
            self.assertGreaterEqual(receiver.incoming.received, count)
            self.assertTrue(sender.done)
            self.assertEqual((dest / source.name).read_bytes(), content)

    def test_confirmed_progress_and_verification_stage(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "first.bin"
            source.write_bytes(np.random.default_rng(8).bytes(p.BLOCK * 4))
            empty = root / "empty"
            empty.write_bytes(b"")
            sender = Sender([source, empty], 0)
            sender.next_file(0)
            sender.handle((visual.CONTROL_PROFILE, (p.STATUS, sender.session, 0,
                          p.encode_json({"state": "receiving", "base": 0, "mask": 12, "received": 2}))), 2)
            self.assertIn("File 50.0%", sender.label())
            self.assertIn("queue 40.0%", sender.label())
            self.assertNotIn("\n", sender.label())
            dest = root / "dest"
            dest.mkdir()
            receiver = Receiver(dest)
            receiver.session = sender.session
            receiver.incoming = p.Incoming(sender.meta)
            receiver.state = "receiving"
            for index in range(4):
                receiver.incoming.add(index, sender.content[index * p.BLOCK:(index + 1) * p.BLOCK])
            receiver.state = "verifying"
            receiver.pending.append((sender.session, receiver.incoming))
            receiver.frame()
            self.assertEqual(receiver.state, "verifying")
            self.assertFalse((dest / source.name).exists())
            receiver.frame()
            self.assertEqual(receiver.state, "complete")
            self.assertEqual((dest / source.name).read_bytes(), source.read_bytes())

    def test_auto_camera_avoids_high_resolution_low_frame_rate(self):
        class Capture:
            def __init__(self):
                self.width, self.height = 640, 480
                self.elapsed = 0

            def set(self, key, value):
                if key == cv2.CAP_PROP_FRAME_WIDTH:
                    self.width = value
                elif key == cv2.CAP_PROP_FRAME_HEIGHT:
                    self.height = value
                return True

            def read(self):
                rate = {640: 30, 1280: 20, 1920: 5}[self.width]
                self.elapsed += 1 / rate
                return True, np.zeros((self.height, self.width, 3), np.uint8)

        capture = Capture()
        app.configure_camera(capture, clock=lambda: capture.elapsed)
        self.assertEqual((capture.width, capture.height), (1280, 720))

    def test_small_footprint_transfers_with_fragmented_metadata_and_drops(self):
        for no_ack in (False, True):
            with self.subTest(no_ack=no_ack), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                source, dest = root / ("long-name-" + "x" * 60 + ".bin"), root / "dest"
                dest.mkdir()
                content = np.random.default_rng(10).bytes(2100)
                source.write_bytes(content)
                sender = (NoAckSender([source]) if no_ack else
                          Sender([source], 0, max_profile=visual.CONTROL_PROFILE))
                receiver = Receiver(dest, no_ack=no_ack)
                rng = np.random.default_rng(35)
                for tick in range(150):
                    now = tick * 0.7
                    profile, data = sender.frame(now)
                    if tick % 5:
                        frame = simulation.distort(visual.render(data, profile, (1280, 720)),
                                                   "Medium", rng, (320, 240), True, True, tick % 4)
                        try:
                            decoded = visual.decode(frame)
                        except ValueError:
                            decoded = None
                        if decoded is not None:
                            receiver.handle(decoded)
                    if no_ack:
                        receiver.frame()
                    if not no_ack and tick % 7:
                        profile, data = receiver.frame()
                        frame = simulation.distort(visual.render(data, profile, (1280, 720)),
                                                   "Medium", rng, (320, 240), True, True, tick % 4)
                        try:
                            decoded = visual.decode(frame)
                        except ValueError:
                            decoded = None
                        if decoded is not None:
                            sender.handle(decoded, now)
                    if (no_ack and receiver.saved) or (not no_ack and sender.done):
                        break
                self.assertEqual((dest / source.name).read_bytes(), content)
                if not no_ack:
                    self.assertTrue(sender.done)
                    self.assertEqual(sender.profile, visual.CONTROL_PROFILE)

    def test_metadata_fragments_out_of_order_duplicates_and_limits(self):
        meta = {"name": "x" * 200, "size": 0, "length": 0, "compressed": False,
                "sha256": "0" * 64, "block": 128}
        first, second = metadata_fragment(meta, 0), metadata_fragment(meta, 1)
        buffer = MetadataBuffer()
        self.assertIsNone(buffer.add(b"12345678", second))
        self.assertIsNone(buffer.add(b"12345678", second))
        self.assertEqual(p.decode_json(buffer.add(b"12345678", first)), meta)
        with self.assertRaises(ValueError):
            buffer.add(b"12345678", b"\0\0\xff\xffbad")
        buffer.add(b"12345678", first)
        with self.assertRaises(ValueError):
            buffer.add(b"12345678", first[:-1] + bytes([first[-1] ^ 1]))

    def test_error_status_fits_coarse_control_frame(self):
        with tempfile.TemporaryDirectory() as temp:
            receiver = Receiver(temp)
            receiver.state = "error"
            receiver.message = "\u753b" * 200
            profile, packet = receiver.frame()
            self.assertEqual(visual.decode(visual.render(packet, profile, (1280, 720))), (profile, p.unpack(packet)))

    def test_camera_automatically_corrects_mirroring_for_sender_and_receiver(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "empty.txt"
            source.write_bytes(b"")
            for mirror in (False, True):
                for role in ("sender", "receiver"):
                    with self.subTest(mirror=mirror, role=role):
                        if role == "sender":
                            machine = Sender([source], time.monotonic())
                            data = p.packet(p.STATUS, machine.session, 0, p.encode_json({"state": "hello"}))
                        else:
                            dest = root / f"dest-{mirror}"
                            dest.mkdir()
                            machine = Receiver(dest, no_ack=True)
                            meta, _ = p.prepare(source)
                            data = p.packet(NO_ACK_META, b"12345678", 0, p.encode_json(meta))
                        image = visual.render(data, 0, (1280, 720))
                        if mirror:
                            image = cv2.flip(image, 1)
                        with patch.object(app, "Camera") as camera, \
                             patch.object(app, "OpticalWindow"), \
                             patch.object(app.cv2, "namedWindow"), \
                             patch.object(app.cv2, "setWindowProperty"), \
                             patch.object(app.cv2, "imshow"), \
                             patch.object(app.cv2, "waitKey", return_value=27), \
                             patch.object(app.cv2, "destroyAllWindows"):
                            camera.return_value.latest.return_value = (image, time.monotonic())
                            app.run_link(machine, 0, (1280, 720), (1280, 720))
                        if role == "sender":
                            self.assertEqual(machine.state, "cal")
                        else:
                            self.assertEqual((dest / source.name).read_bytes(), b"")

    def test_no_ack_sender_does_not_open_camera(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "file"
            path.write_bytes(b"test")
            sender = NoAckSender([path])
            with patch.object(app, "Camera", side_effect=AssertionError("Camera opened")), \
                 patch.object(app, "OpticalWindow"), \
                 patch.object(app.cv2, "namedWindow"), \
                 patch.object(app.cv2, "setWindowProperty"), \
                 patch.object(app.cv2, "imshow"), \
                 patch.object(app.cv2, "waitKey", return_value=27), \
                 patch.object(app.cv2, "destroyAllWindows"):
                result = app.run_link(sender, 0, (1280, 720), (1280, 720))
            self.assertIn("cannot be confirmed", result)

    def test_debug_runner_completes_with_simulated_images(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "file"
            path.write_bytes(os.urandom(600))
            with patch("scrtsf.session.CAL_SECONDS", 0.15), \
                 patch.object(debug_app, "OpticalWindow") as window, \
                 patch.object(app.cv2, "namedWindow"), \
                 patch.object(app.cv2, "setWindowProperty"), \
                 patch.object(app.cv2, "resizeWindow"), \
                 patch.object(app.cv2, "imshow"), \
                 patch.object(app.cv2, "waitKey", return_value=-1), \
                 patch.object(app.cv2, "destroyAllWindows"):
                window.return_value.size = (1280, 720)
                window.return_value.closed.return_value = False
                result = debug_app.run_debug([path], "Light", (1280, 720), footprint=(480, 360), reflected=True)
            self.assertIn("Debug complete: 1 file(s) verified", result)

    def test_no_ack_queue_late_join_drops_and_repeats(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, dest = root / "source", root / "dest"
            source.mkdir()
            dest.mkdir()
            files = [source / "first.bin", source / "empty", source / "second.bin"]
            contents = [os.urandom(11000), b"", os.urandom(9000)]
            for path, content in zip(files, contents):
                path.write_bytes(content)
            sender = NoAckSender(files, 1)
            receiver = Receiver(dest, no_ack=True)
            for tick in range(1000):
                profile, data = sender.frame(tick * 0.4)
                if tick >= 20 and tick % 4:
                    receiver.handle((profile, p.unpack(data)))
                receiver.frame()
                if len(receiver.saved) == len(files) and sender.cycle >= 4:
                    break
            self.assertEqual(receiver.state, "waiting")
            self.assertEqual(len(receiver.saved), len(files))
            for path, content in zip(files, contents):
                self.assertEqual((dest / path.name).read_bytes(), content)

    def test_no_ack_through_distorted_optical_frames(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, dest = root / "file.bin", root / "dest"
            dest.mkdir()
            content = os.urandom(1800)
            source.write_bytes(content)
            sender = NoAckSender([source])
            receiver = Receiver(dest, no_ack=True)
            rng = np.random.default_rng(42)
            for tick in range(30):
                profile, data = sender.frame(tick * 0.4)
                image = visual.render(data, profile, (1280, 720))
                receiver.handle(visual.decode(simulation.distort(image, "Medium", rng)))
                receiver.frame()
                if receiver.saved:
                    break
            self.assertEqual((dest / source.name).read_bytes(), content)

    def test_no_ack_rejects_duplicate_names_and_existing_destination(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            first, second, dest = root / "a", root / "b", root / "dest"
            for folder in (first, second, dest):
                folder.mkdir()
            (first / "same.txt").write_bytes(b"first")
            (second / "same.txt").write_bytes(b"second")
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                NoAckSender([first / "same.txt", second / "same.txt"])
            (dest / "same.txt").write_bytes(b"existing")
            sender = NoAckSender([first / "same.txt"])
            receiver = Receiver(dest, no_ack=True)
            for tick in range(10):
                profile, data = sender.frame(tick)
                receiver.handle((profile, p.unpack(data)))
                receiver.frame()
                if receiver.state == "error":
                    break
            self.assertEqual(receiver.state, "error")
            self.assertEqual((dest / "same.txt").read_bytes(), b"existing")

    def test_debug_loopback_frames_and_acks_use_distortion(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, dest = root / "file", root / "dest"
            dest.mkdir()
            content = os.urandom(1200)
            source.write_bytes(content)
            sender = Sender([source], 0)
            receiver = Receiver(dest)
            rng = np.random.default_rng(5)
            with patch("scrtsf.session.CAL_SECONDS", 1.2):
                for tick in range(20 * len(sender.profile_order)):
                    now = tick * 0.4
                    profile, data = sender.frame(now)
                    image = simulation.distort(visual.render(data, profile, (1280, 720)), "Light", rng)
                    try:
                        receiver.handle(visual.decode(image))
                    except ValueError:
                        pass
                    profile, data = receiver.frame()
                    image = simulation.distort(visual.render(data, profile, (1280, 720)), "Light", rng)
                    try:
                        sender.handle(visual.decode(image), now)
                    except ValueError:
                        pass
                    if sender.done:
                        break
            self.assertTrue(sender.done, sender.label())
            self.assertEqual((dest / source.name).read_bytes(), content)

    def test_exact_limit_with_frame_loss_and_rejects_larger_file(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "limit.bin"
            destination = root / "received"
            destination.mkdir()
            content = os.urandom(p.MAX_FILE)
            source.write_bytes(content)
            sender = Sender([source], 0)
            receiver = Receiver(destination)
            for tick in range(12000):
                now = tick * 0.4
                profile, data = sender.frame(now)
                if tick % 7:
                    receiver.handle((profile, p.unpack(data)))
                if tick % 5:
                    profile, ack = receiver.frame()
                    sender.handle((profile, p.unpack(ack)), now)
                if sender.done:
                    break
            self.assertTrue(sender.done, sender.label())
            self.assertEqual((destination / source.name).read_bytes(), content)
            source.write_bytes(content + b"x")
            with self.assertRaises(ValueError):
                Sender([source], 0)

    def test_full_duplex_through_rendered_frames(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "input.bin"
            destination = root / "received"
            destination.mkdir()
            content = os.urandom(2400)
            source.write_bytes(content)
            sender = Sender([source], 0)
            receiver = Receiver(destination)
            with patch("scrtsf.session.CAL_SECONDS", 1.2):
                for tick in range(150):
                    now = tick * 0.4
                    profile, data = sender.frame(now)
                    if tick % 7:
                        receiver.handle(visual.decode(visual.render(data, profile, (1280, 720))))
                    profile, ack = receiver.frame()
                    if tick % 9:
                        sender.handle(visual.decode(visual.render(ack, profile, (1280, 720))), now)
                    if sender.done:
                        break
            self.assertTrue(sender.done, sender.label())
            self.assertEqual((destination / source.name).read_bytes(), content)

    def test_files_transfer_with_dropped_frames_and_feedback(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source"
            dest = root / "dest"
            source.mkdir()
            dest.mkdir()
            files = [source / "random.bin", source / "empty.txt", source / "unicode-\u65e5.txt"]
            contents = [os.urandom(13500), b"", b"compressible text " * 1000]
            for path, content in zip(files, contents):
                path.write_bytes(content)
            sender = Sender(files, 0)
            receiver = Receiver(dest)
            for tick in range(3000):
                now = tick * 0.4
                profile, data = sender.frame(now)
                if tick % 4 != 0:
                    receiver.handle((profile, p.unpack(data)))
                if tick % 5 != 0:
                    ack_profile, ack = receiver.frame()
                    sender.handle((ack_profile, p.unpack(ack)), now)
                if sender.done:
                    break
            self.assertTrue(sender.done, sender.label())
            self.assertEqual(receiver.state, "end")
            for path, content in zip(files, contents):
                self.assertEqual((dest / path.name).read_bytes(), content)

    def test_refuses_overwrite_and_reports_failure(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            original = root / "a.txt"
            original.write_bytes(b"existing")
            receiver = Receiver(root)
            receiver.session = b"12345678"
            receiver.state = "cal"
            meta = {"name": "a.txt", "size": 0, "length": 0, "compressed": False,
                    "sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"}
            receiver.handle((0, (p.META, b"87654321", 0, p.encode_json(meta))))
            self.assertEqual(receiver.state, "error")
            self.assertEqual(original.read_bytes(), b"existing")
            sender = Sender([original], 0)
            sender.session = receiver.session
            with self.assertRaisesRegex(RuntimeError, "Receiver:"):
                sender.handle((0, p.unpack(receiver.frame()[1])), 1)

    def test_unsafe_names_and_bad_checksum_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            meta = {"name": "..\\bad.txt", "size": 0, "length": 0, "compressed": False,
                    "sha256": "0" * 64}
            with self.assertRaises(ValueError):
                p.Incoming(meta)
            meta["name"] = "safe.txt"
            with self.assertRaisesRegex(ValueError, "checksum"):
                p.Incoming(meta).save(Path(temp))


if __name__ == "__main__":
    unittest.main()
