import math
from unittest.mock import MagicMock

import pytest
import torch
from PIL import Image

from diffusion.callbacks.image_generation import ImageGenerationCallback


@pytest.fixture
def mock_trainer():
    """Create a mock trainer."""
    trainer = MagicMock()
    trainer.current_epoch = 0
    trainer.loggers = None
    return trainer


@pytest.fixture
def mock_pl_module():
    """Create a mock PyTorch Lightning module."""
    module = MagicMock()
    module.device = torch.device("cpu")
    module.sample_timesteps = 10

    def sample_generator(batch, timesteps):
        for i in range(timesteps):
            noise_level = 1.0 - (i / timesteps)
            yield torch.randn_like(batch) * noise_level + torch.ones_like(batch) * (1 - noise_level) * 0.5

    module.sample = sample_generator
    module.eval = MagicMock()
    return module


@pytest.fixture
def sample_batch():
    """Create a sample batch of images."""
    images = torch.randn(4, 3, 32, 32)
    return images, None


def test_init_with_path_string(tmp_path):
    """Test initialization with string path."""
    callback = ImageGenerationCallback(predict_samples=4, output_dir=str(tmp_path), every_n_epochs=1)

    assert callback.shape == (4, 0, 0, 0)
    assert callback.output_dir == tmp_path
    assert callback.every_n_epochs == 1


def test_init_with_path_object(tmp_path):
    """Test initialization with Path object."""
    callback = ImageGenerationCallback(predict_samples=8, output_dir=tmp_path, every_n_epochs=5)

    assert callback.shape == (8, 0, 0, 0)
    assert callback.output_dir == tmp_path
    assert callback.every_n_epochs == 5


def test_init_default_every_n_epochs(tmp_path):
    """Test that every_n_epochs defaults to 1."""
    callback = ImageGenerationCallback(predict_samples=4, output_dir=tmp_path)

    assert callback.every_n_epochs == 1


def test_captures_shape_from_batch(tmp_path, mock_trainer, mock_pl_module, sample_batch):
    """Test that callback captures correct shape from batch."""
    callback = ImageGenerationCallback(predict_samples=4, output_dir=tmp_path)

    assert callback.shape == (4, 0, 0, 0)

    callback.on_validation_batch_start(mock_trainer, mock_pl_module, sample_batch, batch_idx=0, dataloader_idx=0)

    images, _ = sample_batch
    assert callback.shape == (4, 3, 32, 32)
    assert callback.shape[0] == 4
    assert callback.shape[1] == images.shape[1]
    assert callback.shape[2] == images.shape[2]
    assert callback.shape[3] == images.shape[3]


def test_generates_images_on_correct_epoch(tmp_path, mock_trainer, mock_pl_module):
    """Test that images are generated on the correct epochs."""
    callback = ImageGenerationCallback(predict_samples=4, output_dir=tmp_path, every_n_epochs=2)
    callback.shape = (4, 3, 32, 32)

    mock_trainer.current_epoch = 0
    callback.on_validation_epoch_end(mock_trainer, mock_pl_module)
    assert not (tmp_path / "images" / "sample_0.png").exists()

    mock_trainer.current_epoch = 1
    callback.on_validation_epoch_end(mock_trainer, mock_pl_module)
    assert (tmp_path / "images" / "sample_1.png").exists()

    mock_trainer.current_epoch = 2
    callback.on_validation_epoch_end(mock_trainer, mock_pl_module)
    assert not (tmp_path / "images" / "sample_2.png").exists()

    mock_trainer.current_epoch = 3
    callback.on_validation_epoch_end(mock_trainer, mock_pl_module)
    assert (tmp_path / "images" / "sample_3.png").exists()


def test_creates_output_directory(tmp_path, mock_trainer, mock_pl_module):
    """Test that output/images directory is created if it doesn't exist."""
    output_dir = tmp_path / "test_output"
    output_dir.mkdir()
    callback = ImageGenerationCallback(predict_samples=4, output_dir=output_dir, every_n_epochs=1)
    callback.shape = (4, 3, 32, 32)

    images_dir = output_dir / "images"
    assert not images_dir.exists()

    mock_trainer.current_epoch = 0
    callback.on_validation_epoch_end(mock_trainer, mock_pl_module)

    assert images_dir.exists()
    assert (images_dir / "sample_0.png").exists()


def test_saves_image_with_correct_name(tmp_path, mock_trainer, mock_pl_module):
    """Test that images are saved with correct epoch-based names."""
    callback = ImageGenerationCallback(predict_samples=4, output_dir=tmp_path, every_n_epochs=1)
    callback.shape = (4, 3, 32, 32)

    for epoch in [0, 5, 10, 99]:
        mock_trainer.current_epoch = epoch
        callback.on_validation_epoch_end(mock_trainer, mock_pl_module)

        expected_file = tmp_path / "images" / f"sample_{epoch}.png"
        assert expected_file.exists()

        img = Image.open(expected_file)
        assert img.format == "PNG"


def test_calls_model_eval(tmp_path, mock_trainer, mock_pl_module):
    """Test that model is set to eval mode before sampling."""
    callback = ImageGenerationCallback(predict_samples=4, output_dir=tmp_path, every_n_epochs=1)
    callback.shape = (4, 3, 32, 32)

    mock_trainer.current_epoch = 0
    callback.on_validation_epoch_end(mock_trainer, mock_pl_module)

    mock_pl_module.eval.assert_called_once()


def test_uses_correct_device(tmp_path, mock_trainer, mock_pl_module):
    """Test that tensors are created on the correct device."""
    callback = ImageGenerationCallback(predict_samples=4, output_dir=tmp_path, every_n_epochs=1)
    callback.shape = (4, 3, 32, 32)

    mock_pl_module.device = torch.device("cpu")
    mock_trainer.current_epoch = 0

    captured_batch = None

    def capture_sample(batch, timesteps):
        nonlocal captured_batch
        captured_batch = batch
        for _ in range(timesteps):
            yield batch

    mock_pl_module.sample = capture_sample

    callback.on_validation_epoch_end(mock_trainer, mock_pl_module)

    assert captured_batch is not None
    assert captured_batch.device == torch.device("cpu")


def test_uses_correct_sample_timesteps(tmp_path, mock_trainer, mock_pl_module):
    """Test that correct number of timesteps is used for sampling."""
    callback = ImageGenerationCallback(predict_samples=4, output_dir=tmp_path, every_n_epochs=1)
    callback.shape = (4, 3, 32, 32)

    expected_timesteps = 50
    mock_pl_module.sample_timesteps = expected_timesteps
    mock_trainer.current_epoch = 0

    call_count = 0

    def counting_sample(batch, timesteps):
        nonlocal call_count
        assert timesteps == expected_timesteps
        for _ in range(timesteps):
            call_count += 1
            yield batch

    mock_pl_module.sample = counting_sample

    callback.on_validation_epoch_end(mock_trainer, mock_pl_module)

    assert call_count == expected_timesteps


def test_handles_no_loggers(tmp_path, mock_trainer, mock_pl_module):
    """Test that callback works correctly when trainer has no loggers."""
    callback = ImageGenerationCallback(predict_samples=4, output_dir=tmp_path, every_n_epochs=1)
    callback.shape = (4, 3, 32, 32)

    mock_trainer.loggers = None
    mock_trainer.current_epoch = 0

    callback.on_validation_epoch_end(mock_trainer, mock_pl_module)

    assert (tmp_path / "images" / "sample_0.png").exists()


def test_handles_empty_loggers_list(tmp_path, mock_trainer, mock_pl_module):
    """Test that callback works correctly when trainer has empty loggers list."""
    callback = ImageGenerationCallback(predict_samples=4, output_dir=tmp_path, every_n_epochs=1)
    callback.shape = (4, 3, 32, 32)

    mock_trainer.loggers = []
    mock_trainer.current_epoch = 0

    callback.on_validation_epoch_end(mock_trainer, mock_pl_module)

    assert (tmp_path / "images" / "sample_0.png").exists()


def test_generated_image_dimensions(tmp_path, mock_trainer, mock_pl_module):
    """Test that generated images have correct dimensions."""
    callback = ImageGenerationCallback(predict_samples=4, output_dir=tmp_path, every_n_epochs=1)
    callback.shape = (4, 3, 32, 32)

    mock_trainer.current_epoch = 0
    callback.on_validation_epoch_end(mock_trainer, mock_pl_module)

    img_path = tmp_path / "images" / "sample_0.png"
    img = Image.open(img_path)

    image_size = 32
    grid_size = 2
    expected_width = image_size * grid_size
    expected_height = image_size * grid_size

    assert img.size == (expected_width, expected_height)


def test_integration_full_workflow(tmp_path, mock_trainer, mock_pl_module):
    """Test complete workflow: batch start -> epoch end."""
    callback = ImageGenerationCallback(predict_samples=4, output_dir=tmp_path, every_n_epochs=2)

    batch = (torch.randn(8, 3, 28, 28), torch.tensor(list(range(8))))
    callback.on_validation_batch_start(mock_trainer, mock_pl_module, batch, batch_idx=0, dataloader_idx=0)

    assert callback.shape == (4, 3, 28, 28)

    mock_trainer.current_epoch = 1
    callback.on_validation_epoch_end(mock_trainer, mock_pl_module)

    img_path = tmp_path / "images" / "sample_1.png"
    assert img_path.exists()

    img = Image.open(img_path)
    assert img.format == "PNG"
    assert img.size == (56, 56)


def test_integration_multiple_epochs(tmp_path, mock_trainer, mock_pl_module):
    """Test generating images across multiple epochs."""
    callback = ImageGenerationCallback(predict_samples=4, output_dir=tmp_path, every_n_epochs=1)
    callback.shape = (4, 3, 32, 32)

    epochs = [0, 1, 2, 3, 4]
    for epoch in epochs:
        mock_trainer.current_epoch = epoch
        callback.on_validation_epoch_end(mock_trainer, mock_pl_module)

    for epoch in epochs:
        assert (tmp_path / "images" / f"sample_{epoch}.png").exists()


def test_integration_different_sample_counts(tmp_path, mock_trainer, mock_pl_module):
    """Test with different sample counts."""
    for samples in [1, 4, 9, 16]:
        output_dir = tmp_path / f"samples_{samples}"
        output_dir.mkdir()
        callback = ImageGenerationCallback(predict_samples=samples, output_dir=output_dir, every_n_epochs=1)
        callback.shape = (samples, 3, 32, 32)

        mock_trainer.current_epoch = 0
        callback.on_validation_epoch_end(mock_trainer, mock_pl_module)

        img_path = output_dir / "images" / "sample_0.png"
        assert img_path.exists()

        img = Image.open(img_path)
        grid_dim = int(math.sqrt(samples))
        image_size = 32
        expected_size = image_size * grid_dim
        assert img.size == (expected_size, expected_size)
