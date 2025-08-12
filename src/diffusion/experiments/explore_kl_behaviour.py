import plotly.graph_objects as go

from diffusion.diffusion_factors import Factors
from diffusion.schedulers import LinearScheduler
from diffusion.variances import FixedSmallVariance


go


def pprint(tensor):
    print(tensor.squeeze().tolist())


betas = LinearScheduler(1000, 0.0001, 0.02).schedule()
factors = Factors(betas)

pprint(factors.alphas[:5])
pprint(factors.alphas[-5:])

variance_strategy = FixedSmallVariance()
variance = factors.betas * (1 - factors.gammas_prev) / (1 - factors.gammas)
print(len(variance))
print(variance.shape)
print(variance[:5])
print(variance[-5:])

variance = variance.numpy().squeeze()
variance = factors.betas.numpy().squeeze()
variance = variance
x = list(range(len(variance)))
fig = go.Figure()
fig.add_trace(go.Scatter(x=x, y=variance, mode="lines+markers", name="Variance", line=dict(color="blue")))
fig.update(layout=dict(title="Variance over Timesteps", xaxis_title="Timesteps", yaxis_title="Variance"))
fig.show()
