"""Tests for the naming utility module."""

import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from omegaconf import DictConfig, OmegaConf

from diffusion.utils.naming import (
    auto_generate_names,
    generate_run_name,
    validate_naming_config,
    _sanitize_name,
)


@pytest.fixture(autouse=True)
def setup_rank_zero():
    """Setup rank_zero_only.rank for RankedLogger."""
    from lightning_utilities.core.rank_zero import rank_zero_only
    rank_zero_only.rank = 0
    yield
    rank_zero_only.rank = None


class TestSanitizeName:
    """Test name sanitization functionality."""

    def test_lowercase_conversion(self):
        """Test that names are converted to lowercase."""
        assert _sanitize_name("UPPERCASE") == "uppercase"
        assert _sanitize_name("MixedCase") == "mixedcase"

    def test_special_char_replacement(self):
        """Test that special characters are replaced with underscores."""
        assert _sanitize_name("special@chars#here") == "special_chars_here"
        assert _sanitize_name("dots.and.slashes/here") == "dots_and_slashes_here"

    def test_consecutive_underscores(self):
        """Test that consecutive underscores are collapsed."""
        assert _sanitize_name("multiple___underscores") == "multiple_underscores"

    def test_preserves_valid_chars(self):
        """Test that valid characters (alphanumeric, dash, underscore) are preserved."""
        assert _sanitize_name("valid-name_123") == "valid-name_123"

    def test_strips_leading_trailing_underscores(self):
        """Test that leading/trailing underscores are stripped."""
        assert _sanitize_name("_leading") == "leading"
        assert _sanitize_name("trailing_") == "trailing"


class TestGenerateRunName:
    """Test run name generation functionality."""

    def test_basic_generation(self):
        """Test basic run name generation."""
        choices = {
            "diffusion/model": "unet",
            "diffusion/mean_strategy": "epsilon",
            "diffusion/variance_strategy": "fixed_small",
            "diffusion/loss": "mse_epsilon_simple",
        }
        name = generate_run_name(
            choices, "mnist", Path("/tmp/test_logs"), append_timestamp_on_collision=False
        )
        assert name == "unet-epsilon-fixed_small-mse_epsilon_simple"

    def test_generation_with_hybrid_loss(self):
        """Test run name generation with hybrid loss."""
        choices = {
            "diffusion/model": "small_unet",
            "diffusion/mean_strategy": "xstart",
            "diffusion/variance_strategy": "trainable_range",
            "diffusion/loss": "hybrid",
        }
        name = generate_run_name(
            choices, "cifar10", Path("/tmp/test_logs"), append_timestamp_on_collision=False
        )
        assert name == "small_unet-xstart-trainable_range-hybrid"

    def test_collision_handling(self, tmp_path):
        """Test that timestamp is appended on collision."""
        choices = {
            "diffusion/model": "unet",
            "diffusion/mean_strategy": "direct",
            "diffusion/variance_strategy": "trainable_range",
            "diffusion/loss": "hybrid",
        }

        # Create existing directory
        task_dir = tmp_path / "mnist" / "hydra" / "unet-direct-trainable_range-hybrid"
        task_dir.mkdir(parents=True)

        name = generate_run_name(choices, "mnist", tmp_path, append_timestamp_on_collision=True)

        # Should have timestamp appended
        assert name.startswith("unet-direct-trainable_range-hybrid-")
        assert len(name) > len("unet-direct-trainable_range-hybrid-")
        # Check that the suffix is numeric (timestamp)
        suffix = name.split("-")[-1]
        assert suffix.isdigit()
        assert len(suffix) == 14  # YYYYMMDDHHmmss

    def test_no_collision_handling_when_disabled(self, tmp_path):
        """Test that no timestamp is appended when collision handling is disabled."""
        choices = {
            "diffusion/model": "unet",
            "diffusion/mean_strategy": "epsilon",
            "diffusion/variance_strategy": "fixed_large",
            "diffusion/loss": "vlb",
        }

        # Create existing directory
        task_dir = tmp_path / "mnist" / "hydra" / "unet-epsilon-fixed_large-vlb"
        task_dir.mkdir(parents=True)

        name = generate_run_name(choices, "mnist", tmp_path, append_timestamp_on_collision=False)

        # Should not have timestamp appended
        assert name == "unet-epsilon-fixed_large-vlb"


class TestValidateNamingConfig:
    """Test naming configuration validation."""

    def test_valid_config(self):
        """Test that valid config passes validation."""
        cfg = OmegaConf.create({
            "task_name": "mnist",
            "run_name": "valid-run-name",
        })
        # Should not raise any exception
        validate_naming_config(cfg)

    def test_empty_task_name(self):
        """Test that empty task_name raises error."""
        cfg = OmegaConf.create({
            "task_name": None,
            "run_name": "valid-run-name",
        })
        with pytest.raises(ValueError, match="task_name cannot be empty"):
            validate_naming_config(cfg)

    def test_empty_run_name(self):
        """Test that empty run_name raises error."""
        cfg = OmegaConf.create({
            "task_name": "mnist",
            "run_name": None,
        })
        with pytest.raises(ValueError, match="run_name cannot be empty"):
            validate_naming_config(cfg)

    def test_invalid_chars_in_run_name(self):
        """Test that invalid filesystem characters raise error."""
        invalid_names = [
            "run:name",
            "run/name",
            "run\\name",
            'run"name',
            "run*name",
        ]
        for invalid_name in invalid_names:
            cfg = OmegaConf.create({
                "task_name": "mnist",
                "run_name": invalid_name,
            })
            with pytest.raises(ValueError, match="invalid filesystem characters"):
                validate_naming_config(cfg)


class TestAutoGenerateNames:
    """Test auto-generation of names."""

    @patch("diffusion.utils.naming.HydraConfig")
    def test_auto_generate_both_names(self, mock_hydra_config):
        """Test auto-generation of both task_name and run_name."""
        # Mock Hydra config
        mock_runtime = MagicMock()
        mock_runtime.choices = {
            "data": "mnist",
            "diffusion/model": "unet",
            "diffusion/mean_strategy": "epsilon",
            "diffusion/variance_strategy": "fixed_small",
            "diffusion/loss": "mse_epsilon_simple",
        }
        mock_hydra_config.get.return_value.runtime = mock_runtime

        cfg = OmegaConf.create({
            "task_name": None,
            "run_name": None,
            "paths": {"log_dir": "/tmp/logs"},
        })

        result = auto_generate_names(cfg)

        assert result.task_name == "mnist"
        assert result.run_name == "unet-epsilon-fixed_small-mse_epsilon_simple"

    @patch("diffusion.utils.naming.HydraConfig")
    def test_respects_explicit_names(self, mock_hydra_config):
        """Test that explicit names are not overwritten."""
        # Mock Hydra config
        mock_runtime = MagicMock()
        mock_runtime.choices = {
            "data": "mnist",
            "diffusion/model": "unet",
            "diffusion/loss": "hybrid",
            "diffusion/variance_strategy": "trainable_range",
        }
        mock_hydra_config.get.return_value.runtime = mock_runtime

        cfg = OmegaConf.create({
            "task_name": "custom_task",
            "run_name": "custom_run",
            "paths": {"log_dir": "/tmp/logs"},
        })

        result = auto_generate_names(cfg)

        assert result.task_name == "custom_task"
        assert result.run_name == "custom_run"

    @patch("diffusion.utils.naming.HydraConfig")
    def test_debug_mode(self, mock_hydra_config):
        """Test debug mode sets run_name to 'debug' and forces offline."""
        # Mock Hydra config
        mock_runtime = MagicMock()
        mock_runtime.choices = {
            "data": "mnist",
            "diffusion/model": "unet",
            "diffusion/loss": "vlb",
            "diffusion/variance_strategy": "direct",
        }
        mock_hydra_config.get.return_value.runtime = mock_runtime

        cfg = OmegaConf.create({
            "task_name": None,
            "run_name": None,
            "debug_mode": True,
            "paths": {"log_dir": "/tmp/logs"},
            "logger": {
                "wandb": {
                    "offline": False,
                }
            },
        })

        result = auto_generate_names(cfg)

        assert result.run_name == "debug"
        assert result.task_name == "mnist"
        assert result.logger.wandb.offline is True

    @patch("diffusion.utils.naming.HydraConfig")
    def test_debug_mode_respects_explicit_run_name(self, mock_hydra_config):
        """Test debug mode respects explicit run_name if provided."""
        # Mock Hydra config
        mock_runtime = MagicMock()
        mock_runtime.choices = {"data": "mnist"}
        mock_hydra_config.get.return_value.runtime = mock_runtime

        cfg = OmegaConf.create({
            "task_name": "mnist",
            "run_name": "my_debug_run",
            "debug_mode": True,
            "paths": {"log_dir": "/tmp/logs"},
            "logger": {"wandb": {"offline": False}},
        })

        result = auto_generate_names(cfg)

        # Explicit run_name should be preserved even in debug mode
        assert result.run_name == "my_debug_run"
        assert result.logger.wandb.offline is True
