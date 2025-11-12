from dataclasses import dataclass
from typing import Union

from lightning.pytorch.callbacks import RichProgressBar
from lightning.pytorch.callbacks.progress.rich_progress import RichProgressBarTheme
from rich.style import Style


@dataclass
class RichTheme(RichProgressBarTheme):
    description: Union[str, Style] = "bright_cyan"
    progress_bar: Union[str, Style] = "bright_cyan"
    progress_bar_finished: Union[str, Style] = "bright_cyan"
    progress_bar_pulse: Union[str, Style] = "bright_cyan"
    batch_progress: Union[str, Style] = "bright_cyan"
    time: Union[str, Style] = "green"
    processing_speed: Union[str, Style] = "bright_blue"
    metrics: Union[str, Style] = "yellow"
    metrics_text_delimiter: str = " | "
    metrics_format: str = ".8f"


class CustomProgressBar(RichProgressBar):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, theme=RichTheme(), **kwargs)

    def get_metrics(self, trainer, pl_module):
        items = super().get_metrics(trainer, pl_module)
        items.pop("v_num", None)
        return items
