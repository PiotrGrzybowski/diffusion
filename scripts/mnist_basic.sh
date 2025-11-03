
uv run python src/diffusion/scripts/train.py \
    trainer=cpu \
    trainer.max_epochs=20 \
    diffusion/model=unet \
    diffusion/mean_strategy=epsilon \
    diffusion/variance_strategy=fixed_small \
    diffusion/loss=mse_mean_epsilon_simple \
    logger=tensorboard \
    data=mnist \
    dim=28 \
    in_channels=1 \
    out_channels=1 \
    predict_samples=16 \
    task_name=mnist \
    batch_size=32 \
    run_name="epsilon-fixed_small-mse"
