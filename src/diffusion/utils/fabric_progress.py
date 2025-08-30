from contextlib import contextmanager
from typing import Dict, Optional

from lightning import Fabric
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from .ranked_logger import RankedLogger


class FabricProgressLogger:
    """A DDP-friendly progress logger that combines Rich progress bars with ranked logging."""

    def __init__(
        self,
        fabric: Fabric,
        name: str = __name__,
        show_progress: Optional[bool] = None,
        log_interval: int = 10,
    ):
        """
        Initialize the Fabric progress logger.

        Args:
            fabric: Lightning Fabric instance
            name: Logger name
            show_progress: Whether to show progress bars. If None, shows only on rank 0
            log_interval: Interval for progress logging (every N steps)
        """
        self.fabric = fabric
        self.log = RankedLogger(name, rank_zero_only=True)
        self.show_progress = show_progress if show_progress is not None else (fabric.global_rank == 0)
        self.log_interval = log_interval

        # Rich progress columns
        self.columns = (
            TextColumn("[bold]{task.fields[title]}"),
            BarColumn(),
            MofNCompleteColumn(),
            TextColumn("• {task.percentage:>5.1f}%"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
        )

        self._progress: Optional[Progress] = None
        self._tasks: Dict[str, TaskID] = {}

    @contextmanager
    def progress_context(self):
        """Context manager for progress tracking."""
        if self.show_progress:
            self._progress = Progress(*self.columns)
            with self._progress as progress:
                self._progress = progress
                try:
                    yield self
                finally:
                    self._progress = None
                    self._tasks.clear()
        else:
            self._progress = None
            try:
                yield self
            finally:
                self._tasks.clear()

    def add_task(
        self,
        name: str,
        title: str,
        total: int,
        description: Optional[str] = None,
    ) -> str:
        """
        Add a progress task.

        Args:
            name: Unique name for the task
            title: Display title for the task
            total: Total number of steps
            description: Optional description for logging

        Returns:
            Task name for future updates
        """
        if description:
            self.log.info(f"Starting {description}: {title}")
        else:
            self.log.info(f"Starting: {title}")

        if self._progress is not None:
            task_id = self._progress.add_task("", total=total, title=title)
            self._tasks[name] = task_id

        return name

    def update_task(
        self,
        name: str,
        advance: int = 1,
        title: Optional[str] = None,
        log_message: Optional[str] = None,
        force_log: bool = False,
    ) -> None:
        """
        Update a progress task.

        Args:
            name: Task name to update
            advance: Number of steps to advance
            title: New title (optional)
            log_message: Optional log message
            force_log: Force logging regardless of interval
        """
        if name in self._tasks and self._progress is not None:
            task_id = self._tasks[name]

            # Update progress bar
            update_kwargs = {"advance": advance}
            if title is not None:
                update_kwargs["title"] = title

            self._progress.update(task_id, **update_kwargs)

            # Log progress at intervals or when forced
            if force_log or (self._progress.tasks[task_id].completed % self.log_interval == 0):
                task = self._progress.tasks[task_id]
                completed = task.completed
                total = task.total
                percentage = (completed / total * 100) if total > 0 else 0

                if log_message:
                    self.log.info(f"{log_message} [{completed}/{total}] ({percentage:.1f}%)")
                else:
                    task_title = task.fields.get("title", name)
                    self.log.info(f"{task_title}: [{completed}/{total}] ({percentage:.1f}%)")

        elif log_message:
            # Log even without progress bars
            self.log.info(log_message)

    def reset_task(self, name: str, total: int, title: Optional[str] = None) -> None:
        """Reset a task's progress."""
        if name in self._tasks and self._progress is not None:
            task_id = self._tasks[name]
            reset_kwargs = {"total": total}
            if title is not None:
                reset_kwargs["title"] = title
            self._progress.reset(task_id, **reset_kwargs)

    def finish_task(self, name: str, message: Optional[str] = None) -> None:
        """Mark a task as finished."""
        if message:
            self.log.info(f"Finished: {message}")

        if name in self._tasks and self._progress is not None:
            task_id = self._tasks[name]
            task = self._progress.tasks[task_id]
            if not message:
                task_title = task.fields.get("title", name)
                self.log.info(f"Finished: {task_title}")

    def log_info(self, message: str, rank: Optional[int] = None) -> None:
        """Log an info message."""
        self.log.info(message, rank=rank)

    def log_warning(self, message: str, rank: Optional[int] = None) -> None:
        """Log a warning message."""
        self.log.warning(message, rank=rank)

    def log_error(self, message: str, rank: Optional[int] = None) -> None:
        """Log an error message."""
        self.log.error(message, rank=rank)


class NestedProgressTracker:
    """Simplified progress tracker for common nested loop patterns."""

    def __init__(self, fabric: Fabric, name: str = "progress", **logger_kwargs):
        self.logger = FabricProgressLogger(fabric, name, **logger_kwargs)
        self._context = None

    def __enter__(self):
        self._context = self.logger.progress_context()
        return self._context.__enter__()

    def __exit__(self, *args):
        return self._context.__exit__(*args)

    def setup_sampling(self, batches: int, timesteps: int) -> "SamplingProgress":
        """Setup progress tracking for sampling loops."""
        return SamplingProgress(self.logger, batches, timesteps)

    def setup_training(self, epochs: int, batches_per_epoch: int, timesteps_per_batch: int = None) -> "TrainingProgress":
        """Setup progress tracking for training loops."""
        return TrainingProgress(self.logger, epochs, batches_per_epoch, timesteps_per_batch)


class SamplingProgress:
    """Encapsulates sampling progress tracking."""

    def __init__(self, logger: FabricProgressLogger, batches: int, timesteps: int):
        self.logger = logger
        self.batches = batches
        self.timesteps = timesteps
        self.current_batch = 0

        # Setup tasks
        self.logger.add_task("overall", "Overall Sampling", batches * timesteps, "Generating samples")
        self.logger.add_task("batches", "Batches", batches)
        self.logger.add_task("timesteps", f"Timesteps (Batch 1/{batches})", timesteps)

    def next_batch(self):
        """Move to next batch."""
        self.current_batch += 1
        self.logger.update_task("batches", advance=1)

        if self.current_batch <= self.batches:
            self.logger.reset_task("timesteps", self.timesteps, f"Timesteps (Batch {self.current_batch}/{self.batches})")

    def step(self, log_message: str = None):
        """Advance one timestep."""
        self.logger.update_task("timesteps", advance=1, log_message=log_message)
        self.logger.update_task("overall", advance=1)

    def finish(self, message: str = "Sampling completed!"):
        """Finish sampling."""
        self.logger.finish_task("overall", message)


class TrainingProgress:
    """Encapsulates training progress tracking."""

    def __init__(self, logger: FabricProgressLogger, epochs: int, batches_per_epoch: int, timesteps_per_batch: int = None):
        self.logger = logger
        self.epochs = epochs
        self.batches_per_epoch = batches_per_epoch
        self.timesteps_per_batch = timesteps_per_batch
        self.current_epoch = 0
        self.current_batch = 0

        # Calculate total steps
        total_steps = epochs * batches_per_epoch
        if timesteps_per_batch:
            total_steps *= timesteps_per_batch

        # Setup tasks
        self.logger.add_task("overall", "Overall Training", total_steps, "Training model")
        self.logger.add_task("epochs", "Epochs", epochs)
        self.logger.add_task("batches", f"Batches (Epoch 1/{epochs})", batches_per_epoch)

        if timesteps_per_batch:
            self.logger.add_task("timesteps", "Timesteps (Epoch 1, Batch 1)", timesteps_per_batch)

    def next_epoch(self):
        """Move to next epoch."""
        self.current_epoch += 1
        self.current_batch = 0
        self.logger.update_task("epochs", advance=1, force_log=True)
        self.logger.log_info(f"Starting epoch {self.current_epoch}/{self.epochs}")

        self.logger.reset_task("batches", self.batches_per_epoch, f"Batches (Epoch {self.current_epoch}/{self.epochs})")

        if self.timesteps_per_batch:
            self.logger.reset_task("timesteps", self.timesteps_per_batch, f"Timesteps (Epoch {self.current_epoch}, Batch 1)")

    def next_batch(self):
        """Move to next batch."""
        self.current_batch += 1
        self.logger.update_task("batches", advance=1)

        if self.timesteps_per_batch and self.current_batch <= self.batches_per_epoch:
            self.logger.reset_task(
                "timesteps", self.timesteps_per_batch, f"Timesteps (Epoch {self.current_epoch}, Batch {self.current_batch})"
            )

    def step(self, log_message: str = None):
        """Advance one step (timestep if configured)."""
        if self.timesteps_per_batch:
            self.logger.update_task("timesteps", advance=1, log_message=log_message)
        self.logger.update_task("overall", advance=1)

    def log_batch_completion(self):
        """Log batch completion."""
        if self.current_batch % 5 == 0:
            self.logger.log_info(f"Completed batch {self.current_batch}/{self.batches_per_epoch} in epoch {self.current_epoch}")

    def finish(self, message: str = "Training completed successfully!"):
        """Finish training."""
        self.logger.finish_task("overall", message)


# Convenience function for quick setup
def create_fabric_progress_logger(fabric: Fabric, name: str = __name__, **kwargs) -> FabricProgressLogger:
    """Create a fabric progress logger instance."""
    return FabricProgressLogger(fabric, name, **kwargs)


def create_nested_progress_tracker(fabric: Fabric, name: str = "progress", **kwargs) -> NestedProgressTracker:
    """Create a simplified nested progress tracker."""
    return NestedProgressTracker(fabric, name, **kwargs)

