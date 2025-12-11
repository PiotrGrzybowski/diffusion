from dataclasses import dataclass

from lightning.pytorch.callbacks import RichProgressBar
from lightning.pytorch.callbacks.progress.rich_progress import RichProgressBarTheme
from rich.style import Style


@dataclass
class RichTheme(RichProgressBarTheme):
    description: str | Style = "bright_cyan"
    progress_bar: str | Style = "bright_cyan"
    progress_bar_finished: str | Style = "bright_cyan"
    progress_bar_pulse: str | Style = "bright_cyan"
    batch_progress: str | Style = "bright_cyan"
    time: str | Style = "green"
    processing_speed: str | Style = "bright_blue"
    metrics: str | Style = "yellow"
    metrics_text_delimiter: str = " | "
    metrics_format: str = ".8f"


class CustomProgressBar(RichProgressBar):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, theme=RichTheme(), **kwargs)

    def get_metrics(self, trainer, pl_module):
        items = super().get_metrics(trainer, pl_module)
        items.pop("v_num", None)
        return items
