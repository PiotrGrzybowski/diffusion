import torch
from torchmetrics.image.fid import FrechetInceptionDistance
from tqdm import tqdm


device = "cuda:0"
# device = "cpu"

fid = FrechetInceptionDistance(feature=64).to(device)

for _ in tqdm(range(100)):
    # generate two slightly overlapping image intensity distributions
    imgs_dist1 = torch.randint(0, 200, (100, 3, 299, 299), dtype=torch.uint8, device=device)
    imgs_dist2 = torch.randint(100, 255, (100, 3, 299, 299), dtype=torch.uint8, device=device)
    fid.update(imgs_dist1, real=True)
    fid.update(imgs_dist2, real=False)
    score = fid.compute()
