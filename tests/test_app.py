import base64
from io import BytesIO
from pathlib import Path
import re
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image
import torch
from torchvision.transforms import functional as TF, InterpolationMode

from app import create_app
from inference import InvalidImage, Segmenter, decode_image, preprocess
from model import PetUNet


def photo(mode="RGB", size=(19, 11), color=None, format="PNG"):
    stream = BytesIO()
    Image.new(mode, size, color).save(stream, format=format)
    stream.seek(0)
    return stream


class PreprocessingTests(unittest.TestCase):
    def test_matches_original_predict_pipeline_on_pattern(self):
        # Nonuniform, nonsquare RGB data detects channel, scaling, and resize changes.
        image = Image.new("RGB", (31, 17))
        image.putdata([(i % 256, (i * 7) % 256, (i * 13) % 256) for i in range(31 * 17)])
        expected = TF.to_tensor(TF.resize(image, [64, 64], InterpolationMode.BILINEAR))
        expected = TF.normalize(expected, [0.485, 0.456, 0.406],
                                [0.229, 0.224, 0.225]).unsqueeze(0)
        actual = preprocess(image, 64)
        self.assertTrue(torch.equal(actual, expected))
        self.assertEqual(tuple(actual.shape), (1, 3, 64, 64))
        self.assertEqual(actual.dtype, torch.float32)
        self.assertEqual(actual.device.type, "cpu")

    def test_rgb_order_scaling_and_normalization(self):
        actual = preprocess(Image.new("RGB", (3, 5), (255, 128, 0)), 64)
        expected = torch.tensor([(1 - .485) / .229, (128 / 255 - .456) / .224,
                                 (0 - .406) / .225])
        torch.testing.assert_close(actual[0, :, 20, 20], expected)

    def test_grayscale_and_rgba_decode_as_rgb(self):
        for mode in ("L", "RGBA"):
            with self.subTest(mode=mode):
                image = decode_image(photo(mode))
                self.assertEqual(image.mode, "RGB")
                self.assertEqual(preprocess(image, 96).shape, (1, 3, 96, 96))

    def test_rejects_decoded_pixel_limit_before_conversion(self):
        with patch("inference.MAX_IMAGE_PIXELS", 10):
            with self.assertRaisesRegex(InvalidImage, "8 megapixels"):
                decode_image(photo())


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.directory = tempfile.TemporaryDirectory()
        cls.checkpoint_path = Path(cls.directory.name) / "synthetic.pt"
        torch.manual_seed(42)
        # A real state dictionary; random weights test compatibility, not accuracy.
        torch.save({"model": PetUNet(pretrained=False).state_dict(),
                    "config": {"size": 64}}, cls.checkpoint_path)
        cls.segmenter = Segmenter(cls.checkpoint_path)

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def setUp(self):
        with patch("app.Segmenter", return_value=self.segmenter) as loader:
            self.app = create_app(str(self.checkpoint_path))
        self.assertEqual(loader.call_count, 1)
        self.client = self.app.test_client()

    def test_real_checkpoint_cpu_eval_and_forward(self):
        self.assertEqual(self.segmenter.size, 64)
        self.assertFalse(self.segmenter.model.training)
        self.assertEqual(next(self.segmenter.model.parameters()).device.type, "cpu")
        result = self.segmenter.predict(photo(color=(10, 120, 240)))
        self.assertEqual(result.mask.size, (19, 11))
        self.assertEqual(result.mask.mode, "L")
        self.assertTrue(set(result.mask.tobytes()) <= {0, 255})

    def test_postprocessing_resizes_logits_before_sigmoid_and_threshold(self):
        # At the center, interpolated logits are 0: >= must include this pixel.
        class KnownLogits(torch.nn.Module):
            def forward(inner, tensor):
                self.assertTrue(torch.is_inference_mode_enabled())
                self.assertEqual(tensor.shape, (1, 3, 64, 64))
                return torch.tensor([[[[-4., 4.], [-4., 4.]]]])
        with patch.object(self.segmenter, "model", KnownLogits()):
            result = self.segmenter.predict(photo(size=(5, 3)))
        self.assertEqual(list(result.mask.tobytes()), [0, 0, 255, 255, 255] * 3)
        self.assertEqual(result.overlay.size, (5, 3))

        # Asymmetric logits distinguish resizing logits from resizing probabilities.
        class AsymmetricLogits(torch.nn.Module):
            def forward(inner, tensor):
                return torch.tensor([[[[-4., 1.], [-4., 1.]]]])
        with patch.object(self.segmenter, "model", AsymmetricLogits()):
            result = self.segmenter.predict(photo(size=(9, 3)))
        self.assertEqual(list(result.mask.tobytes()), ([0] * 6 + [255] * 3) * 3)

    def test_successful_upload_uses_shared_pipeline_and_loads_no_more_models(self):
        with patch("app.Segmenter") as constructor, patch(
            "inference.preprocess", wraps=preprocess
        ) as shared:
            for _ in range(2):
                response = self.client.post("/segment", data={"image": (photo(), "pet.png")})
                self.assertEqual(response.status_code, 200)
                self.assertIn(b"Pet overlay", response.data)
                encoded = re.search(rb'data:image/png;base64,([^\"]+)', response.data).group(1)
                with Image.open(BytesIO(base64.b64decode(encoded))) as mask:
                    self.assertEqual(mask.size, (19, 11))
                    self.assertEqual(mask.mode, "L")
            self.assertEqual(shared.call_count, 2)
            constructor.assert_not_called()

    def test_missing_empty_invalid_unsupported_and_truncated_uploads(self):
        truncated = photo(size=(200, 200)).getvalue()[:50]
        cases = [None, (BytesIO(b""), ""), (BytesIO(b""), "empty.png"),
                 (BytesIO(b"not a photo"), "fake.jpg"),
                 (photo(format="BMP"), "fake.png"),
                 (BytesIO(truncated), "broken.png")]
        for item in cases:
            with self.subTest(item=item):
                data = {} if item is None else {"image": item}
                response = self.client.post("/segment", data=data)
                self.assertEqual(response.status_code, 400)
                self.assertIn(b'role="alert"', response.data)
        self.assertEqual(self.client.get("/").status_code, 200)
        self.assertEqual(self.client.post("/segment", data={"image": (photo(), "ok.png")}).status_code, 200)

    def test_oversized_upload(self):
        self.app.config["MAX_CONTENT_LENGTH"] = 1024
        response = self.client.post("/segment", data={"image": (BytesIO(b"a" * 2048), "big.jpg")})
        self.assertEqual(response.status_code, 413)
        self.assertIn(b"10 MiB", response.data)

    def test_pixel_limit_and_animation_upload_errors(self):
        with patch("inference.MAX_IMAGE_PIXELS", 10):
            response = self.client.post("/segment", data={"image": (photo(), "big.png")})
        self.assertEqual(response.status_code, 400)
        self.assertIn(b"8 megapixels", response.data)
        animation = BytesIO()
        Image.new("RGB", (10, 10), "red").save(
            animation, format="PNG", save_all=True,
            append_images=[Image.new("RGB", (10, 10), "blue")], duration=100,
        )
        animation.seek(0)
        response = self.client.post("/segment", data={"image": (animation, "animated.png")})
        self.assertEqual(response.status_code, 400)
        self.assertIn(b"still image", response.data)

    def test_busy_request_and_recovery(self):
        lock = self.app.extensions["prediction_lock"]
        with lock:
            response = self.client.post("/segment", data={"image": (photo(), "pet.png")})
            self.assertEqual(response.status_code, 503)
            self.assertEqual(self.client.get("/").status_code, 200)
        self.assertEqual(self.client.post("/segment", data={"image": (photo(), "pet.png")}).status_code, 200)

    def test_inference_error_does_not_leave_server_busy(self):
        with patch.object(self.segmenter, "predict", side_effect=RuntimeError("internal detail")):
            with self.assertLogs(self.app.logger, level="ERROR"):
                response = self.client.post("/segment", data={"image": (photo(), "pet.png")})
        self.assertEqual(response.status_code, 500)
        self.assertNotIn(b"internal detail", response.data)
        self.assertFalse(self.app.extensions["prediction_lock"].locked())

    def test_missing_checkpoint_configuration_and_file(self):
        with patch.dict("os.environ", {}, clear=True):
            with self.assertRaisesRegex(ValueError, "PET_CHECKPOINT"):
                create_app()
        with self.assertRaisesRegex(ValueError, "does not exist"):
            Segmenter(Path(self.directory.name) / "missing.pt")

    def test_environment_checkpoint_path(self):
        with patch.dict("os.environ", {"PET_CHECKPOINT": str(self.checkpoint_path)}):
            with patch("app.Segmenter", return_value=self.segmenter) as loader:
                create_app()
                loader.assert_called_once_with(str(self.checkpoint_path))

    def test_checkpoint_requires_config_size_and_strict_architecture(self):
        cases = [{}, {"model": {}, "config": {}},
                 {"model": {}, "config": {"size": "256"}},
                 {"model": {}, "config": {"size": True}},
                 {"model": {}, "config": {"size": 1024}},
                 {"model": {}, "config": {"size": 64}}]
        for checkpoint in cases:
            with self.subTest(checkpoint=checkpoint), patch("inference.torch.load", return_value=checkpoint) as load:
                with self.assertRaisesRegex(ValueError, "Cannot load PetUNet checkpoint"):
                    Segmenter(self.checkpoint_path)
                load.assert_called_once_with(self.checkpoint_path, map_location="cpu", weights_only=True)

    def test_corrupt_checkpoint(self):
        path = Path(self.directory.name) / "corrupt.pt"
        path.write_bytes(b"not a checkpoint")
        with self.assertRaisesRegex(ValueError, "Cannot load PetUNet checkpoint"):
            Segmenter(path)


if __name__ == "__main__":
    unittest.main()
