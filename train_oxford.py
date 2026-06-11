# /// script
# requires-python = ">=3.9"
# dependencies = [
#     "datasets",
#     "fire",
#     "torch",
#     "torchvision",
# ]
# ///

import fire
import torch
import torchvision.transforms as T

from torch.utils.data import Dataset
from datasets import load_dataset

from rin_pytorch import RIN, GaussianDiffusion, Trainer

# dataset

class OxfordFlowersDataset(Dataset):
    def __init__(
        self,
        image_size
    ):
        self.ds = load_dataset('nelorth/oxford-flowers')['train']

        self.transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.PILToTensor()
        ])

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        pil = self.ds[idx]['image']
        tensor = self.transform(pil)
        return tensor / 255.

# main training script

def train(
    inverted_cross_attn: bool = True
):
    dataset = OxfordFlowersDataset(
        image_size = 64
    )

    model = RIN(
        dim = 256,
        image_size = 64,
        patch_size = 8,
        depth = 6,
        num_latents = 128,
        latent_self_attn_depth = 2,
        inverted_cross_attn = inverted_cross_attn
    )

    diffusion = GaussianDiffusion(
        model,
        timesteps = 400,
        train_prob_self_cond = 0.9,
        scale = 1.
    )

    trainer = Trainer(
        diffusion,
        dataset = dataset,
        train_batch_size = 16,
        train_num_steps = 70_000,
        results_folder = './results'
    )

    trainer.train()

if __name__ == '__main__':
    fire.Fire(train)
