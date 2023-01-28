import math
from pathlib import Path
from random import random
from functools import partial
from multiprocessing import cpu_count

import torch
from torch import nn, einsum
from torch.special import expm1
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from torch.optim import Adam
from torchvision import transforms as T, utils

from beartype import beartype

from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange

from PIL import Image
from tqdm.auto import tqdm
from ema_pytorch import EMA

from accelerate import Accelerator

# helpers functions

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def divisible_by(numer, denom):
    return (numer % denom) == 0

def cycle(dl):
    while True:
        for data in dl:
            yield data

def has_int_squareroot(num):
    return (math.sqrt(num) ** 2) == num

def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

def convert_image_to(img_type, image):
    if image.mode != img_type:
        return image.convert(img_type)
    return image

# use layernorm without bias, more stable

class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.register_buffer("beta", torch.zeros(dim))

    def forward(self, x):
        return F.layer_norm(x, x.shape[-1:], self.gamma, self.beta)

# positional embeds

class LearnedSinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim))

    def forward(self, x):
        x = rearrange(x, 'b -> b 1')
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim = -1)
        fouriered = torch.cat((x, fouriered), dim = -1)
        return fouriered

class LinearAttention(nn.Module):
    def __init__(
        self,
        dim,
        heads = 4,
        dim_head = 32,
        norm = False,
        time_cond_dim = None
    ):
        super().__init__()
        hidden_dim = dim_head * heads
        self.scale = dim_head ** -0.5
        self.heads = heads

        self.time_cond = None

        if exists(time_cond_dim):
            self.time_cond = nn.Sequential(
                nn.SiLU(),
                nn.Linear(time_cond_dim, dim * 2),
                Rearrange('b d -> b 1 d')
            )

            nn.init.zeros_(self.time_cond[-2].weight)
            nn.init.zeros_(self.time_cond[-2].bias)

        self.norm = LayerNorm(dim) if norm else nn.Identity()

        self.to_qkv = nn.Linear(dim, hidden_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(hidden_dim, dim, bias = False),
            LayerNorm(dim)
        )

    def forward(
        self,
        x,
        time = None
    ):
        h = self.heads
        x = self.norm(x)

        if exists(self.time_cond):
            assert exists(time)
            scale, shift = self.time_cond(time).chunk(2, dim = -1)
            x = (x * (scale + 1)) + shift

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        q = q.softmax(dim = -1)
        k = k.softmax(dim = -2)

        q = q * self.scale

        context = torch.einsum('b h n d, b h n e -> b h d e', k, v)

        out = torch.einsum('b h d e, b h n d -> b h n e', context, q)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Attention(nn.Module):
    def __init__(
        self,
        dim,
        dim_context = None,
        heads = 4,
        dim_head = 32,
        norm = False,
        norm_context = False,
        time_cond_dim = None
    ):
        super().__init__()
        hidden_dim = dim_head * heads
        dim_context = default(dim_context, dim)

        self.time_cond = None

        if exists(time_cond_dim):
            self.time_cond = nn.Sequential(
                nn.SiLU(),
                nn.Linear(time_cond_dim, dim * 2),
                Rearrange('b d -> b 1 d')
            )

            nn.init.zeros_(self.time_cond[-2].weight)
            nn.init.zeros_(self.time_cond[-2].bias)

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.norm = LayerNorm(dim) if norm else nn.Identity()
        self.norm_context = LayerNorm(dim_context) if norm_context else nn.Identity()

        self.to_q = nn.Linear(dim, hidden_dim, bias = False)
        self.to_kv = nn.Linear(dim_context, hidden_dim * 2, bias = False)
        self.to_out = nn.Linear(hidden_dim, dim, bias = False)

    def forward(
        self,
        x,
        context = None,
        time = None
    ):
        h = self.heads
        has_context = exists(context)

        context = default(context, x)

        if x.shape[-1] != self.norm.gamma.shape[-1]:
            print(context.shape, x.shape, self.norm.gamma.shape)

        x = self.norm(x)

        if exists(self.time_cond):
            assert exists(time)
            scale, shift = self.time_cond(time).chunk(2, dim = -1)
            x = (x * (scale + 1)) + shift

        if has_context:
            context = self.norm_context(context)

        qkv = (self.to_q(x), *self.to_kv(context).chunk(2, dim = -1))
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        q = q * self.scale

        sim = einsum('b h i d, b h j d -> b h i j', q, k)
        attn = sim.softmax(dim = -1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class PEG(nn.Module):
    def __init__(
        self,
        dim
    ):
        super().__init__()
        self.ds_conv = nn.Conv2d(dim, dim, 3, padding = 1, groups = dim)

    def forward(self, x):
        b, n, d = x.shape
        hw = int(math.sqrt(n))
        x = rearrange(x, 'b (h w) d -> b d h w', h = hw)
        x = self.ds_conv(x)
        x = rearrange(x, 'b d h w -> b (h w) d')
        return x

class FeedForward(nn.Module):
    def __init__(self, dim, mult = 4, time_cond_dim = None):
        super().__init__()
        self.norm = LayerNorm(dim)

        self.time_cond = None

        if exists(time_cond_dim):
            self.time_cond = nn.Sequential(
                nn.SiLU(),
                nn.Linear(time_cond_dim, dim * 2),
                Rearrange('b d -> b 1 d')
            )

            nn.init.zeros_(self.time_cond[-2].weight)
            nn.init.zeros_(self.time_cond[-2].bias)

        inner_dim = int(dim * mult)
        self.net = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU(),
            nn.Linear(inner_dim, dim)
        )

    def forward(self, x, time = None):
        x = self.norm(x)

        if exists(self.time_cond):
            assert exists(time)
            scale, shift = self.time_cond(time).chunk(2, dim = -1)
            x = (x * (scale + 1)) + shift

        return self.net(x)

# model

class RINBlock(nn.Module):
    def __init__(
        self,
        dim,
        latent_self_attn_depth,
        dim_latent = None,
        final_norm = True,
        **attn_kwargs
    ):
        super().__init__()
        dim_latent = default(dim_latent, dim)

        self.latents_attend_to_patches = Attention(dim_latent, dim_context = dim, norm = True, norm_context = True, **attn_kwargs)

        self.latent_self_attns = nn.ModuleList([])
        for _ in range(latent_self_attn_depth):
            self.latent_self_attns.append(nn.ModuleList([
                Attention(dim_latent, norm = True, **attn_kwargs),
                FeedForward(dim_latent)
            ]))

        self.latent_final_norm = LayerNorm(dim_latent) if final_norm else nn.Identity()

        self.patches_peg = PEG(dim)
        self.patches_self_attn = LinearAttention(dim, norm = True, **attn_kwargs)
        self.patches_self_attn_ff = FeedForward(dim)

        self.patches_attend_to_latents = Attention(dim, dim_context = dim_latent, norm = True, norm_context = True, **attn_kwargs)
        self.patches_cross_attn_ff = FeedForward(dim)

    def forward(self, patches, latents, t):
        patches = self.patches_peg(patches) + patches

        # latents extract or cluster information from the patches

        latents = self.latents_attend_to_patches(latents, patches, time = t) + latents

        # latent self attention

        for attn, ff in self.latent_self_attns:
            latents = attn(latents, time = t) + latents
            latents = ff(latents, time = t) + latents

        # additional patches self attention with linear attention

        patches = self.patches_self_attn(patches, time = t) + patches
        patches = self.patches_self_attn_ff(patches) + patches

        # patches attend to the latents

        patches = self.patches_attend_to_latents(patches, latents, time = t) + patches

        patches = self.patches_cross_attn_ff(patches, time = t) + patches

        latents = self.latent_final_norm(latents)
        return patches, latents

class RIN(nn.Module):
    def __init__(
        self,
        dim,
        image_size,
        patch_size = 16,
        channels = 3,
        depth = 6,                      # number of RIN blocks
        latent_self_attn_depth = 2,     # how many self attentions for the latent per each round of cross attending from pixel space to latents and back
        dim_latent = None,              # will default to image dim (dim)
        num_latents = 256,              # they still had to use a fair amount of latents for good results (256), in line with the Perceiver line of papers from Deepmind
        learned_sinusoidal_dim = 16,
        **attn_kwargs
    ):
        super().__init__()
        assert divisible_by(image_size, patch_size)
        dim_latent = default(dim_latent, dim)

        self.channels = channels # times 2 due to self-conditioning

        patch_height_width = image_size // patch_size
        num_patches = patch_height_width ** 2
        pixel_patch_dim = channels * (patch_size ** 2)

        # time conditioning

        sinu_pos_emb = LearnedSinusoidalPosEmb(learned_sinusoidal_dim)
        time_dim = dim * 4
        fourier_dim = learned_sinusoidal_dim + 1

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        # pixels to patch and back

        self.to_patches = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (c p1 p2)', p1 = patch_size, p2 = patch_size),
            nn.Linear(pixel_patch_dim * 2, dim)
        )

        self.axial_pos_emb = nn.Parameter(torch.randn(2, patch_height_width, dim) * 0.02)

        self.to_pixels = nn.Sequential(
            LayerNorm(dim),
            nn.Linear(dim, pixel_patch_dim),
            Rearrange('b (h w) (c p1 p2) -> b c (h p1) (w p2)', p1 = patch_size, p2 = patch_size, h = patch_height_width)
        )

        self.latents = nn.Parameter(torch.randn(num_latents, dim_latent))
        nn.init.normal_(self.latents, std = 0.02)

        self.init_self_cond_latents = nn.Sequential(
            FeedForward(dim_latent),
            LayerNorm(dim_latent)
        )

        nn.init.zeros_(self.init_self_cond_latents[-1].gamma)

        # the main RIN body parameters  - another attention is all you need moment

        attn_kwargs = {**attn_kwargs, 'time_cond_dim': time_dim}

        self.blocks = nn.ModuleList([RINBlock(dim, dim_latent = dim_latent, latent_self_attn_depth = latent_self_attn_depth, **attn_kwargs) for _ in range(depth)])

    def forward(
        self,
        x,
        time,
        x_self_cond = None,
        latent_self_cond = None,
        return_latents = False
    ):
        batch = x.shape[0]

        x_self_cond = default(x_self_cond, lambda: torch.zeros_like(x))

        x = torch.cat((x_self_cond, x), dim = 1)

        # prepare time conditioning

        t = self.time_mlp(time)

        # prepare latents

        latents = repeat(self.latents, 'n d -> b n d', b = batch)

        # the warm starting of latents as in the paper

        if exists(latent_self_cond):
            latents = latents + self.init_self_cond_latents(latent_self_cond)

        # to patches

        patches = self.to_patches(x)

        pos_emb_h, pos_emb_w = self.axial_pos_emb
        pos_emb = rearrange(pos_emb_h, 'i d -> i 1 d') + rearrange(pos_emb_w, 'j d -> 1 j d')
        patches = patches + rearrange(pos_emb, 'i j d -> (i j) d')

        # the recurrent interface network body

        for block in self.blocks:
            patches, latents = block(patches, latents, t)

        # to pixels

        pixels = self.to_pixels(patches)

        if not return_latents:
            return pixels

        return pixels, latents

# normalize and unnormalize image

def normalize_img(x):
    return x * 2 - 1

def unnormalize_img(x):
    return (x + 1) * 0.5

# helper functions

def log(t, eps = 1e-20):
    return torch.log(t.clamp(min = eps))

def right_pad_dims_to(x, t):
    padding_dims = x.ndim - t.ndim
    if padding_dims <= 0:
        return t
    return t.view(*t.shape, *((1,) * padding_dims))

def beta_linear_log_snr(t):
    return -log(expm1(1e-4 + 10 * (t ** 2)))

def alpha_cosine_log_snr(t, s = 0.008):
    return -log((torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** -2) - 1, eps = 1e-5)

def gamma_sigmoid_log_snr(t, start = -3, end = 3, tau = 1, clamp_min = 1e-5):
    v_start = torch.tensor(start / tau).sigmoid()
    v_end = torch.tensor(end / tau).sigmoid()
    gamma = (-((t * (end - start) + start) / tau).sigmoid() + v_end) / (v_end - v_start)
    gamma.clamp_(min = clamp_min, max = 1.)
    return -log(gamma ** -1. - 1, eps = 1e-5)

def log_snr_to_alpha_sigma(log_snr):
    return torch.sqrt(torch.sigmoid(log_snr)), torch.sqrt(torch.sigmoid(-log_snr))

# gaussian diffusion

@beartype
class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        model: RIN,
        *,
        image_size,
        timesteps = 1000,
        use_ddim = False,
        noise_schedule = 'sigmoid',
        objective = 'eps',
        schedule_kwargs: dict = dict(),
        time_difference = 0.,
        train_prob_self_cond = 0.9
    ):
        super().__init__()
        self.model = model
        self.channels = self.model.channels

        assert objective in {'x0', 'eps'}, 'objective must be either predict x0 or noise'
        self.objective = objective

        self.image_size = image_size

        if noise_schedule == "linear":
            self.log_snr = beta_linear_log_snr
        elif noise_schedule == "cosine":
            self.log_snr = alpha_cosine_log_snr
        elif noise_schedule == "sigmoid":
            self.log_snr = gamma_sigmoid_log_snr
        else:
            raise ValueError(f'invalid noise schedule {noise_schedule}')

        self.log_snr = partial(self.log_snr, **schedule_kwargs)

        self.timesteps = timesteps
        self.use_ddim = use_ddim

        # proposed in the paper, summed to time_next
        # as a way to fix a deficiency in self-conditioning and lower FID when the number of sampling timesteps is < 400

        self.time_difference = time_difference

        # probability for self conditioning during training

        self.train_prob_self_cond = train_prob_self_cond

    @property
    def device(self):
        return next(self.model.parameters()).device

    def get_sampling_timesteps(self, batch, *, device):
        times = torch.linspace(1., 0., self.timesteps + 1, device = device)
        times = repeat(times, 't -> b t', b = batch)
        times = torch.stack((times[:, :-1], times[:, 1:]), dim = 0)
        times = times.unbind(dim = -1)
        return times

    @torch.no_grad()
    def ddpm_sample(self, shape, time_difference = None):
        batch, device = shape[0], self.device

        time_difference = default(time_difference, self.time_difference)

        time_pairs = self.get_sampling_timesteps(batch, device = device)

        img = torch.randn(shape, device=device)

        x_start = None
        last_latents = None

        for time, time_next in tqdm(time_pairs, desc = 'sampling loop time step', total = self.timesteps):

            # add the time delay

            time_next = (time_next - self.time_difference).clamp(min = 0.)

            noise_cond = self.log_snr(time)

            # get predicted x0

            model_output, last_latents = self.model(img, noise_cond, x_start, last_latents, return_latents = True)

            # get log(snr)

            log_snr = self.log_snr(time)
            log_snr_next = self.log_snr(time_next)
            log_snr, log_snr_next = map(partial(right_pad_dims_to, img), (log_snr, log_snr_next))

            # get alpha sigma of time and next time

            alpha, sigma = log_snr_to_alpha_sigma(log_snr)
            alpha_next, sigma_next = log_snr_to_alpha_sigma(log_snr_next)

            # calculate x0 and noise

            if self.objective == 'x0':
                x_start = model_output

            elif self.objective == 'eps':
                x_start = (img - sigma * model_output) / alpha

            # clip x0

            x_start.clamp_(-1., 1.)

            # derive posterior mean and variance

            c = -expm1(log_snr - log_snr_next)

            mean = alpha_next * (img * (1 - c) / alpha + c * x_start)
            variance = (sigma_next ** 2) * c
            log_variance = log(variance)

            # get noise

            noise = torch.where(
                rearrange(time_next > 0, 'b -> b 1 1 1'),
                torch.randn_like(img),
                torch.zeros_like(img)
            )

            img = mean + (0.5 * log_variance).exp() * noise

        return unnormalize_img(img)

    @torch.no_grad()
    def ddim_sample(self, shape, time_difference = None):
        batch, device = shape[0], self.device

        time_difference = default(time_difference, self.time_difference)

        time_pairs = self.get_sampling_timesteps(batch, device = device)

        img = torch.randn(shape, device = device)

        x_start = None
        last_latents = None

        for times, times_next in tqdm(time_pairs, desc = 'sampling loop time step'):

            # get times and noise levels

            log_snr = self.log_snr(times)
            log_snr_next = self.log_snr(times_next)

            padded_log_snr, padded_log_snr_next = map(partial(right_pad_dims_to, img), (log_snr, log_snr_next))

            alpha, sigma = log_snr_to_alpha_sigma(padded_log_snr)
            alpha_next, sigma_next = log_snr_to_alpha_sigma(padded_log_snr_next)

            # add the time delay

            times_next = (times_next - time_difference).clamp(min = 0.)

            # predict x0

            model_output, last_latents = self.model(img, log_snr, x_start, last_latents, return_latents = True)

            # calculate x0 and noise

            if self.objective == 'x0':
                x_start = model_output

            elif self.objective == 'eps':
                x_start = (img - sigma * model_output) / alpha

            # clip x0

            x_start.clamp_(-1., 1.)

            # get predicted noise

            if self.objective == 'x0':
                pred_noise = (img - alpha * x_start) / sigma.clamp(min = 1e-8)

            elif self.objective == 'eps':
                pred_noise = model_output

            # calculate x next

            img = x_start * alpha_next + pred_noise * sigma_next

        return unnormalize_img(img)

    @torch.no_grad()
    def sample(self, batch_size = 16):
        image_size, channels = self.image_size, self.channels
        sample_fn = self.ddpm_sample if not self.use_ddim else self.ddim_sample
        return sample_fn((batch_size, channels, image_size, image_size))

    def forward(self, img, *args, **kwargs):
        batch, c, h, w, device, img_size, = *img.shape, img.device, self.image_size
        assert h == img_size and w == img_size, f'height and width of image must be {img_size}'

        # sample random times

        times = torch.zeros((batch,), device = device).float().uniform_(0, 1.)

        # convert image to bit representation

        img = normalize_img(img)

        # noise sample

        noise = torch.randn_like(img)

        noise_level = self.log_snr(times)
        padded_noise_level = right_pad_dims_to(img, noise_level)
        alpha, sigma =  log_snr_to_alpha_sigma(padded_noise_level)

        noised_img = alpha * img + sigma * noise

        # in the paper, they had to use a really high probability of latent self conditioning, up to 90% of the time
        # slight drawback

        self_cond = self_latents = None

        if random() < self.train_prob_self_cond:
            with torch.no_grad():
                self_cond, self_latents = self.model(noised_img, noise_level, return_latents = True)
                self_cond = self_cond.detach()
                self_latents = self_latents.detach()

        # predict and take gradient step

        pred = self.model(noised_img, noise_level, self_cond, self_latents)

        if self.objective == 'x0':
            target = img

        elif self.objective == 'eps':
            target = noise

        return F.mse_loss(pred, target)

# dataset classes

class Dataset(Dataset):
    def __init__(
        self,
        folder,
        image_size,
        exts = ['jpg', 'jpeg', 'png', 'tiff'],
        augment_horizontal_flip = False,
        convert_image_to = None
    ):
        super().__init__()
        self.folder = folder
        self.image_size = image_size
        self.paths = [p for ext in exts for p in Path(f'{folder}').glob(f'**/*.{ext}')]

        maybe_convert_fn = partial(convert_image_to, convert_image_to) if exists(convert_image_to) else nn.Identity()

        self.transform = T.Compose([
            T.Lambda(maybe_convert_fn),
            T.Resize(image_size),
            T.RandomHorizontalFlip() if augment_horizontal_flip else nn.Identity(),
            T.CenterCrop(image_size),
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(path)
        return self.transform(img)

# trainer class

@beartype
class Trainer(object):
    def __init__(
        self,
        diffusion_model: GaussianDiffusion,
        folder,
        *,
        train_batch_size = 16,
        gradient_accumulate_every = 1,
        augment_horizontal_flip = True,
        train_lr = 1e-4,
        train_num_steps = 100000,
        ema_update_every = 10,
        ema_decay = 0.995,
        adam_betas = (0.9, 0.99),
        save_and_sample_every = 1000,
        num_samples = 25,
        results_folder = './results',
        amp = False,
        fp16 = False,
        split_batches = True,
        convert_image_to = None
    ):
        super().__init__()

        self.accelerator = Accelerator(
            split_batches = split_batches,
            mixed_precision = 'fp16' if fp16 else 'no'
        )

        self.accelerator.native_amp = amp

        self.model = diffusion_model

        assert has_int_squareroot(num_samples), 'number of samples must have an integer square root'
        self.num_samples = num_samples
        self.save_and_sample_every = save_and_sample_every

        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every

        self.train_num_steps = train_num_steps
        self.image_size = diffusion_model.image_size

        # dataset and dataloader

        self.ds = Dataset(folder, self.image_size, augment_horizontal_flip = augment_horizontal_flip, convert_image_to = convert_image_to)
        dl = DataLoader(self.ds, batch_size = train_batch_size, shuffle = True, pin_memory = True, num_workers = cpu_count())

        dl = self.accelerator.prepare(dl)
        self.dl = cycle(dl)

        # optimizer

        self.opt = Adam(diffusion_model.parameters(), lr = train_lr, betas = adam_betas)

        # for logging results in a folder periodically

        if self.accelerator.is_main_process:
            self.ema = EMA(diffusion_model, beta = ema_decay, update_every = ema_update_every)

            self.results_folder = Path(results_folder)
            self.results_folder.mkdir(exist_ok = True)

        # step counter state

        self.step = 0

        # prepare model, dataloader, optimizer with accelerator

        self.model, self.opt = self.accelerator.prepare(self.model, self.opt)

    def save(self, milestone):
        if not self.accelerator.is_local_main_process:
            return

        data = {
            'step': self.step,
            'model': self.accelerator.get_state_dict(self.model),
            'opt': self.opt.state_dict(),
            'ema': self.ema.state_dict(),
            'scaler': self.accelerator.scaler.state_dict() if exists(self.accelerator.scaler) else None
        }

        torch.save(data, str(self.results_folder / f'model-{milestone}.pt'))

    def load(self, milestone):
        data = torch.load(str(self.results_folder / f'model-{milestone}.pt'))

        model = self.accelerator.unwrap_model(self.model)
        model.load_state_dict(data['model'])

        self.step = data['step']
        self.opt.load_state_dict(data['opt'])
        self.ema.load_state_dict(data['ema'])

        if exists(self.accelerator.scaler) and exists(data['scaler']):
            self.accelerator.scaler.load_state_dict(data['scaler'])

    def train(self):
        accelerator = self.accelerator
        device = accelerator.device

        with tqdm(initial = self.step, total = self.train_num_steps, disable = not accelerator.is_main_process) as pbar:

            while self.step < self.train_num_steps:

                total_loss = 0.

                for _ in range(self.gradient_accumulate_every):
                    data = next(self.dl).to(device)

                    with self.accelerator.autocast():
                        loss = self.model(data)
                        loss = loss / self.gradient_accumulate_every
                        total_loss += loss.item()

                    self.accelerator.backward(loss)

                pbar.set_description(f'loss: {total_loss:.4f}')

                accelerator.wait_for_everyone()

                self.opt.step()
                self.opt.zero_grad()

                accelerator.wait_for_everyone()

                if accelerator.is_main_process:
                    self.ema.to(device)
                    self.ema.update()

                    if self.step != 0 and self.step % self.save_and_sample_every == 0:
                        self.ema.ema_model.eval()

                        with torch.no_grad():
                            milestone = self.step // self.save_and_sample_every
                            batches = num_to_groups(self.num_samples, self.batch_size)
                            all_images_list = list(map(lambda n: self.ema.ema_model.sample(batch_size=n), batches))

                        all_images = torch.cat(all_images_list, dim = 0)
                        utils.save_image(all_images, str(self.results_folder / f'sample-{milestone}.png'), nrow = int(math.sqrt(self.num_samples)))
                        self.save(milestone)

                self.step += 1
                pbar.update(1)

        accelerator.print('training complete')
