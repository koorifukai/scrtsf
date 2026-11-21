import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from scrtsf import app, diagnostics
from scrtsf import protocol as p
from scrtsf import simulation, visual
from scrtsf.session import NO_ACK_META, NoAckSender, Receiver, Sender


class OpticalTests(unittest.TestCase):
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
        wrong = diagnostics.inspect_image(loaded)
        self.assertIn("Looking for four markers", wrong)
        report = diagnostics.inspect_image(loaded, mirror=True)
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


class SessionTests(unittest.TestCase):
    def test_camera_mirror_option_for_sender_and_receiver(self):
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
                             patch.object(app.cv2, "namedWindow"), \
                             patch.object(app.cv2, "setWindowProperty"), \
                             patch.object(app.cv2, "imshow"), \
                             patch.object(app.cv2, "waitKey", return_value=27), \
                             patch.object(app.cv2, "destroyAllWindows"):
                            camera.return_value.latest.return_value = (image, time.monotonic())
                            app.run_link(machine, 0, (1280, 720), (1280, 720), mirror)
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
            with patch("scrtsf.session.CAL_SECONDS", 0.1), \
                 patch.object(app.cv2, "namedWindow"), \
                 patch.object(app.cv2, "setWindowProperty"), \
                 patch.object(app.cv2, "resizeWindow"), \
                 patch.object(app.cv2, "imshow"), \
                 patch.object(app.cv2, "waitKey", return_value=-1), \
                 patch.object(app.cv2, "destroyAllWindows"):
                result = app.run_debug([path], "Light", (1280, 720))
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
                for tick in range(100):
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
