import shutil
from pathlib import Path

import fire
import rootutils
from huggingface_hub import list_repo_tree, snapshot_download


root_path = rootutils.setup_root(__file__, indicator="pyproject.toml", pythonpath=False)

HF_REPO = "PiotrGrzybowski/diffusion-model-zoo"
LOGS_DIR = root_path / "logs"


def _is_folder(entry) -> bool:
    return type(entry).__name__ == "RepoFolder"


def _get_tasks() -> list[str]:
    entries = list_repo_tree(HF_REPO, path_in_repo="")
    return sorted(e.path for e in entries if _is_folder(e))


def _get_runs(task: str) -> list[str]:
    entries = list_repo_tree(HF_REPO, path_in_repo=f"{task}/hydra")
    return sorted(e.path.split("/")[-1] for e in entries if _is_folder(e))


def _run_dir(task: str, run: str) -> Path:
    return LOGS_DIR / f"zoo_{task}" / "hydra" / run


class ModelZoo:
    """Download and manage pretrained models from Hugging Face Hub."""

    def list(self, task: str | None = None):
        """List available tasks or runs.

        Args:
            task: If provided, list runs for this task. Otherwise list all tasks.
        """
        if task is None:
            for t in _get_tasks():
                print(f"  {t}")
            return

        runs = _get_runs(task)
        if not runs:
            print(f"No runs found for task: {task}")
            return

        for run in runs:
            local = _run_dir(task, run)
            status = " [downloaded]" if local.exists() else ""
            print(f"  {run}{status}")

    def download(self, task: str, run: str | None = None, force: bool = False):
        """Download a run or all runs for a task.

        Args:
            task: Task name (e.g. cifar10).
            run: Run name. If omitted, downloads all runs.
            force: Re-download even if already present.
        """
        runs = [run] if run else _get_runs(task)

        if not runs:
            print(f"No runs found for task: {task}")
            return

        for run_name in runs:
            local = _run_dir(task, run_name)
            ckpt = local / "checkpoints" / "last.ckpt"

            if ckpt.exists() and not force:
                print(f"Already exists: {local}")
                continue

            print(f"Downloading {task}/{run_name}...")

            cache_dir = root_path / ".hf_download_cache"
            snapshot_download(
                repo_id=HF_REPO,
                allow_patterns=f"{task}/hydra/{run_name}/**",
                local_dir=cache_dir,
            )

            cached = cache_dir / task / "hydra" / run_name
            local.parent.mkdir(parents=True, exist_ok=True)
            if local.exists():
                shutil.rmtree(local)
            shutil.copytree(cached, local)
            shutil.rmtree(cache_dir, ignore_errors=True)

            print(f"Saved to: {local}")

    def delete(self, task: str, run: str | None = None):
        """Delete downloaded run(s) from local disk.

        Args:
            task: Task name.
            run: Run name. If omitted, deletes all downloaded runs for the task.
        """
        if run:
            targets = [_run_dir(task, run)]
        else:
            task_dir = LOGS_DIR / f"zoo_{task}" / "hydra"
            if not task_dir.exists():
                print(f"Nothing downloaded for task: {task}")
                return
            targets = sorted(task_dir.iterdir())

        for target in targets:
            if target.exists():
                shutil.rmtree(target)
                print(f"Deleted: {target}")
            else:
                print(f"Not found: {target}")


def main():
    fire.Fire(ModelZoo)


if __name__ == "__main__":
    main()
