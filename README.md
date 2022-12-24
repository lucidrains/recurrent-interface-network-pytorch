<img src="./images/rin.png" width="500png"></img>

<img src="./images/latent-self-conditioning.png" width="600px"></img>

## Recurrent Interface Network (RIN) - Pytorch (wip)

Implementation of <a href="https://arxiv.org/abs/2212.11972">Recurrent Interface Network (RIN)</a>, for highly efficient generation of images and video without cascading networks, in Pytorch. The author unawaredly reinvented the <a href="https://github.com/lucidrains/isab-pytorch">induced set-attention block</a> from the <a href="https://arxiv.org/abs/1810.00825">set transformers</a> paper. They also combine this with the self-conditioning technique from the <a href="https://arxiv.org/abs/2208.04202">Bit Diffusion paper</a>, specifically for the latents. The last ingredient seems to be a new noise function based around the sigmoid, which the author claims is better than cosine scheduler for larger images.

The big surprise is that the generations can reach this level of fidelity. Will need to verify this on my own machine

## Install

```bash
$ pip install rin-pytorch
```

## Usage

```python
from rin_pytorch import RIN, Trainer, GaussianDiffusion

model = RIN(
    dim = 32,
    channels = 3,
    dim_mults = (1, 2, 4, 8),
).cuda()

diffusion = GaussianDiffusion(
    model,
    image_size = 128,
    timesteps = 100,
    use_ddim = True              # use ddim
).cuda()

trainer = Trainer(
    diffusion,
    '/path/to/your/data',             # path to your folder of images
    results_folder = './results',     # where to save results
    num_samples = 16,                 # number of samples
    train_batch_size = 4,             # training batch size
    gradient_accumulate_every = 4,    # gradient accumulation
    train_lr = 1e-4,                  # learning rate
    save_and_sample_every = 1000,     # how often to save and sample
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
    dim = 64,
    dim_mults = (1, 2, 4, 8)
)

diffusion = GaussianDiffusion(
    model,
    image_size = 128,
    timesteps = 1000
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
