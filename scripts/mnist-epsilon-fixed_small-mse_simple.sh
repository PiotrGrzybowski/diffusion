uv run python src/diffusion/scripts/train.py \
    diffusion/model=unet \
    diffusion/mean_strategy=epsilon \
    diffusion/variance_strategy=fixed_small \
    diffusion/loss=mse_mean_epsilon_simple \
    trainer=ddp \
    trainer.max_epochs=20 \
    logger=wandb \
    data=mnist \
    predict_samples=16 \
    task_name=mnist \
    run_name="debug2" \
    timesteps=1000 \
    sample_timesteps=1000 \
    batch_size=128
