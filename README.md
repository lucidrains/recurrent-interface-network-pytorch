<img src="./images/rin.png" width="500png"></img>

<img src="./images/latent-self-conditioning.png" width="600px"></img>

## Recurrent Interface Network (RIN) - Pytorch

Implementation of <a href="https://arxiv.org/abs/2212.11972">Recurrent Interface Network (RIN)</a>, for highly efficient generation of images and video without cascading networks, in Pytorch. The author unawaredly reinvented the <a href="https://github.com/lucidrains/isab-pytorch">induced set-attention block</a> from the <a href="https://arxiv.org/abs/1810.00825">set transformers</a> paper. They also combine this with the self-conditioning technique from the <a href="https://arxiv.org/abs/2208.04202">Bit Diffusion paper</a>, specifically for the latents. The last ingredient seems to be a new noise function based around the sigmoid, which the author claims is better than cosine scheduler for larger images.

The big surprise is that the generations can reach this level of fidelity. Will need to verify this on my own machine

Additionally, we will try adding an extra linear attention on the main branch, in addition to the full self attention on the latents. Self conditioning will also be applied to the non-latent images in pixel-space. Let us see how far we can push this approach.

## Install

```bash
$ pip install rin-pytorch
```

## Usage

```python
from rin_pytorch import GaussianDiffusion, RIN, Trainer

model = RIN(
    dim = 256,                  # model dimensions
    image_size = 128,           # image size
    patch_size = 8,             # patch size
    num_latents = 128,          # number of latents. they used 256 in the paper
    latent_self_attn_depth = 4, # number of latent self attention blocks per recurrent step, K in the paper
).cuda()

diffusion = GaussianDiffusion(
    model,
    image_size = 128,
    use_ddim = False,
    timesteps = 400,
    train_prob_self_cond = 0.9  # how often to self condition on latents
).cuda()

trainer = Trainer(
    diffusion,
    '/home/phil/dl/data/flowers',
    results_folder = './rin',
    num_samples = 16,
    train_batch_size = 4,
    gradient_accumulate_every = 4,
    train_lr = 1e-4,
    save_and_sample_every = 1000,
    train_num_steps = 700000,         # total training steps
    ema_decay = 0.995,                # exponential moving average decay
)

trainer.train()
```

Results will be saved periodically to the `./results` folder

If you would like to experiment with the `RIN` and `GaussianDiffusion` class outside the `Trainer`

```python
import torch
from rin_pytorch import RIN, GaussianDiffusion

model = RIN(
    dim = 256,                  # model dimensions
    image_size = 128,           # image size
    patch_size = 8,             # patch size
    num_latents = 128,          # number of latents. they used 256 in the paper
    latent_self_attn_depth = 4, # number of latent self attention blocks per recurrent step, K in the paper
).cuda()

diffusion = GaussianDiffusion(
    model,
    image_size = 128,
    timesteps = 1000,
    train_prob_self_cond = 0.9
)

training_images = torch.randn(8, 3, 128, 128) # images are normalized from 0 to 1
loss = diffusion(training_images)
loss.backward()
# after a lot of training

sampled_images = diffusion.sample(batch_size = 4)
sampled_images.shape # (4, 3, 128, 128)
```

## Citations

```bibtex
@misc{jabri2022scalable,
    title   = {Scalable Adaptive Computation for Iterative Generation}, 
    author  = {Allan Jabri and David Fleet and Ting Chen},
    year    = {2022},
    eprint  = {2212.11972},
    archivePrefix = {arXiv},
    primaryClass = {cs.LG}
}
```
