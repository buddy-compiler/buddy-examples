from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
from PIL import Image

from calibration import load_calibration_images


class CalibrationTest(unittest.TestCase):
    def test_guest_corner_order_and_padding(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "sample.png"
            Image.fromarray(np.array([[[0, 0, 0], [200, 0, 0]]], dtype=np.uint8)).save(path)
            actual = load_calibration_images([path], 4)
        expected = np.full((1, 3, 4, 4), np.float32(114) / 255, dtype=np.float32)
        expected[0, :, 1:3, :] = 0
        expected[0, 0, 1, :] = np.array([0, 0, 200, 200], dtype=np.float32) / 255
        expected[0, 0, 2, :] = np.array([0, 100, 200, 200], dtype=np.float32) / 255
        np.testing.assert_array_equal(actual, expected)

    def test_actual_format_not_filename_extension(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "sample.png"
            Image.new("RGB", (2, 2), (60, 120, 180)).save(path, format="JPEG")
            with Image.open(path) as image:
                expected = np.asarray(image.convert("RGB"), dtype=np.float32).transpose(2, 0, 1) / 255
            actual = load_calibration_images([path, path], 2)
        np.testing.assert_array_equal(actual, np.stack([expected, expected]))
        self.assertEqual(actual.dtype, np.float32)

    def test_missing_or_empty_input_fails(self):
        with TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                load_calibration_images([Path(directory) / "missing.png"], 4)
        with self.assertRaises(ValueError):
            load_calibration_images([], 4)


if __name__ == "__main__":
    unittest.main()
