import torch


def gaussian_kl(mean_p: torch.Tensor, log_var_p: torch.Tensor, mean_q: torch.Tensor, log_var_q: torch.Tensor) -> torch.Tensor:
    """Compute the KL divergence between two gaussians.
    Args:
        mean_p (torch.Tensor): The mean of the first Gaussian.
        var_p (torch.Tensor): The variance of the first Gaussian.
        mean_q (torch.Tensor): The mean of the second Gaussian.
        var_q (torch.Tensor): The variance of the second Gaussian.

    Returns:
        torch.Tensor: The KL divergence between the two Gaussians.
    """

    return 0.5 * (-1.0 + log_var_q - log_var_p + torch.exp(log_var_p - log_var_q) + ((mean_p - mean_q) ** 2) * torch.exp(-log_var_q))


def approx_standard_normal_cdf(x: torch.Tensor) -> torch.Tensor:
    """A fast approximation of the cumulative distribution function of the standard normal.
    Args:
        x (torch.Tensor): The input tensor to compute the CDF for.

    Returns:
        torch.Tensor: The CDF of the input tensor.
    """
    return 0.5 * (1.0 + torch.tanh(torch.sqrt(torch.tensor(2.0) / torch.pi) * (x + 0.044715 * torch.pow(x, 3))))


def discretized_gaussian_log_likelihood(x_start: torch.Tensor, mean: torch.Tensor, variance: torch.Tensor) -> torch.Tensor:
    """Compute the log-likelihood of a Gaussian distribution discretizing to a given image.
    Args:
        x (torch.Tensor): target images. It is assumed torch.Tensor was uint8 values, rescaled to torch.Tensor range [-1, 1].
        mean (torch.Tensor): Gaussian mean Tensor.
        variance (torch.Tensor): Gaussian variance Tensor.
    Returns:
        torch.Tensor: a tensor like x of log probabilities (in nats).
    """
    step = 1.0 / 255.0
    centered_x = x_start - mean
    std = torch.sqrt(variance)
    upper = (centered_x + step) / std
    lower = (centered_x - step) / std

    cdf_upper = approx_standard_normal_cdf(upper)
    cdf_lower = approx_standard_normal_cdf(lower)
    cdf_delta = cdf_upper - cdf_lower

    likelihood = torch.where(x_start < -0.999, cdf_upper, torch.where(x_start > 0.999, 1 - cdf_lower, cdf_delta))
    return torch.log(likelihood.clamp(min=1e-12))


def prior_kl(mean_p: torch.Tensor, log_var_p: torch.Tensor) -> torch.Tensor:
    """Compute the KL divergence between given and Normal Gaussians"""
    mean_q = torch.zeros_like(mean_p, device=mean_p.device)
    log_var_q = torch.zeros_like(log_var_p, device=mean_p.device)

    return gaussian_kl(mean_p, log_var_p, mean_q, log_var_q)

    # def calc_bpd_loop(self, model, x_start, clip_denoised=True, model_kwargs=None):
    #     """
    #     Compute the entire variational lower-bound, measured in bits-per-dim,
    #     as well as other related quantities.
    #
    #     :param model: the model to evaluate loss on.
    #     :param x_start: the [N x C x ...] tensor of inputs.
    #     :param clip_denoised: if True, clip denoised samples.
    #     :param model_kwargs: if not None, a dict of extra keyword arguments to
    #         pass to the model. This can be used for conditioning.
    #
    #     :return: a dict containing the following keys:
    #              - total_bpd: the total variational lower-bound, per batch element.
    #              - prior_bpd: the prior term in the lower-bound.
    #              - vb: an [N x T] tensor of terms in the lower-bound.
    #              - xstart_mse: an [N x T] tensor of x_0 MSEs for each timestep.
    #              - mse: an [N x T] tensor of epsilon MSEs for each timestep.
    #     """
    #     device = x_start.device
    #     batch_size = x_start.shape[0]
    #
    #     vb = []
    #     xstart_mse = []
    #     mse = []
    #     for t in list(range(self.num_timesteps))[::-1]:
    #         t_batch = th.tensor([t] * batch_size, device=device)
    #         noise = th.randn_like(x_start)
    #         x_t = self.q_sample(x_start=x_start, t=t_batch, noise=noise)
    #         # Calculate VLB term at the current timestep
    #         with th.no_grad():
    #             out = self._vb_terms_bpd(
    #                 model,
    #                 x_start=x_start,
    #                 x_t=x_t,
    #                 t=t_batch,
    #                 clip_denoised=clip_denoised,
    #                 model_kwargs=model_kwargs,
    #             )
    #         vb.append(out["output"])
    #         xstart_mse.append(mean_flat((out["pred_xstart"] - x_start) ** 2))
    #         eps = self._predict_eps_from_xstart(x_t, t_batch, out["pred_xstart"])
    #         mse.append(mean_flat((eps - noise) ** 2))
    #
    #     vb = th.stack(vb, dim=1)
    #     xstart_mse = th.stack(xstart_mse, dim=1)
    #     mse = th.stack(mse, dim=1)
    #
    #     prior_bpd = self._prior_bpd(x_start)
    #     total_bpd = vb.sum(dim=1) + prior_bpd
    #     return {
    #         "total_bpd": total_bpd,
    #         "prior_bpd": prior_bpd,
    #         "vb": vb,
    #         "xstart_mse": xstart_mse,
    #         "mse": mse,
    #     }
    #
