import sys
from pathlib import Path

import rootutils
from huggingface_hub import HfApi


root_path = rootutils.setup_root(__file__, indicator="pyproject.toml", pythonpath=False)

HF_REPO = "PiotrGrzybowski/diffusion-model-zoo"


def upload_task(source_dir: Path, task_name: str, run_names: list[str] | None = None):
    """Upload runs from a local task directory to Hugging Face Hub.

    Expects source_dir to contain: hydra/{run_name}/{config.yaml, checkpoints/, images/, ...}
    Uploads to HF as: {task_name}/{run_name}/...
    """
    hydra_dir = source_dir / "hydra"
    if not hydra_dir.is_dir():
        print(f"No hydra/ directory found in {source_dir}")
        sys.exit(1)

    if run_names:
        runs = [hydra_dir / name for name in run_names]
        missing = [r for r in runs if not r.is_dir()]
        if missing:
            print(f"Run directories not found: {[m.name for m in missing]}")
            sys.exit(1)
    else:
        runs = sorted(p for p in hydra_dir.iterdir() if p.is_dir())

    if not runs:
        print(f"No runs found in {hydra_dir}")
        sys.exit(1)

    api = HfApi()
    api.create_repo(repo_id=HF_REPO, exist_ok=True)

    for run_dir in runs:
        run_name = run_dir.name
        hf_prefix = f"{task_name}/{run_name}"
        print(f"Uploading {run_dir.name} -> {HF_REPO}/{hf_prefix}")

        api.upload_folder(
            repo_id=HF_REPO,
            folder_path=str(run_dir),
            path_in_repo=hf_prefix,
        )
        print(f"  Done: {hf_prefix}")

    print(f"\nAll uploads complete: https://huggingface.co/{HF_REPO}")


def main():
    args = sys.argv[1:]

    if len(args) < 2:
        print("Usage: uv run upload <source_dir> <task_name> [run_name ...]")
        print()
        print("Examples:")
        print("  # Upload all runs from model_zoo/cifar10/")
        print("  uv run upload model_zoo/cifar10 model_zoo_cifar")
        print()
        print("  # Upload specific runs")
        print("  uv run upload model_zoo/cifar10 model_zoo_cifar unet-epsilon-fixed_small-vlb-linear")
        print()
        print("  # Upload from new_logs")
        print("  uv run upload new_logs/my_experiment my_experiment")
        sys.exit(1)

    source_dir = Path(args[0])
    if not source_dir.is_absolute():
        source_dir = root_path / source_dir

    task_name = args[1]
    run_names = args[2:] if len(args) > 2 else None

    upload_task(source_dir, task_name, run_names)


if __name__ == "__main__":
    main()
