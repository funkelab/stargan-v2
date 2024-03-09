"""Reduces the model into just want is needed for inference."""
from os.path import join as ospj
from starganv2.core.model import Generator, MappingNetwork, StyleEncoder
from starganv2.core.checkpoint import CheckpointIO
import torch


class LatentInferenceModel(torch.nn.Module):
    def __init__(
        self, checkpoint_dir, img_size, style_dim, latent_dim, input_dim=1, num_domains=6, w_hpf=0.0
    ) -> None:
        super().__init__()
        generator = Generator(img_size, style_dim, w_hpf=w_hpf, input_dim=input_dim)
        mapping_network = MappingNetwork(latent_dim, style_dim, num_domains=num_domains)

        self.nets = torch.nn.ModuleDict({
            "generator": generator,
            "mapping_network": mapping_network,
        })

        self.checkpoint_io = CheckpointIO(
            ospj(checkpoint_dir, "{:06d}_nets_ema.ckpt"),
            data_parallel=False,
            **self.nets,
        )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.nets.to(self.device)
        self.latent_dim = latent_dim
        self.style_dim = style_dim

    def load_checkpoint(self, step):
        self.checkpoint_io.load(step)
    
    def forward(self, x_src, y_trg):
        z = torch.randn(x_src.size(0), self.latent_dim).to(self.device)
        s = self.nets.mapping_network(z, y_trg)
        x_fake = self.nets.generator(x_src, s)
        return x_fake

# TODO ReferenceInferenceModel