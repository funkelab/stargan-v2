import pytest
from starganv2.core.model import Generator, StyleEncoder, Discriminator
import torch


@pytest.fixture
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # return torch.device("cpu")


@pytest.mark.parametrize(
    "img_size, style_dim, max_conv_dim, input_dim",
    [
        (224, 64, 512, 3),  # Retina size
        (128, 64, 512, 1),  # Synapse size
        (128, 512, 512, 1),  # Synapse size, bigger style
        (32, 64, 512, 7),  # Kidney cell
    ],
)
def test_generator(img_size, style_dim, max_conv_dim, input_dim, device):
    gen = Generator(
        img_size=img_size,
        style_dim=style_dim,
        max_conv_dim=max_conv_dim,
        input_dim=input_dim,
        w_hpf=0,
    )
    input = torch.randn(1, input_dim, img_size, img_size)
    style = torch.randn(1, style_dim)

    gen.to(device)
    output = gen(input.to(device), style.to(device))

    assert output.shape == (1, input_dim, img_size, img_size)


@pytest.mark.parametrize(
    "img_size, style_dim, num_domains, max_conv_dim, input_dim",
    [
        (224, 64, 5, 512, 3),  # Retina size
        (128, 64, 6, 512, 1),  # Synapse size
        (128, 512, 6, 512, 1),  # Synapse size, bigger style
        (32, 64, 8, 512, 7),  # Kidney cell
    ],
)
def test_style_encoder(img_size, style_dim, num_domains, max_conv_dim, input_dim, device):
    encoder = StyleEncoder(
        img_size=img_size,
        style_dim=style_dim,
        num_domains=num_domains,
        max_conv_dim=max_conv_dim,
        input_dim=input_dim,
    )

    input = torch.randn(1, input_dim, img_size, img_size)
    y = torch.randint(0, num_domains, (1,))

    encoder.to(device)
    output = encoder(input.to(device), y.to(device))

    assert output.shape == (1, style_dim)


@pytest.mark.parametrize(
    "img_size, num_domains, max_conv_dim, input_dim",
    [
        (224, 5, 512, 3),  # Retina size
        (128, 6, 512, 1),  # Synapse size
        (32, 8, 512, 7),  # Kidney cell
    ],
)
def test_discriminator(img_size, num_domains, max_conv_dim, input_dim, device):
    disc = Discriminator(
        img_size=img_size,
        num_domains=num_domains,
        max_conv_dim=max_conv_dim,
        input_dim=input_dim,
    )

    input = torch.randn(1, input_dim, img_size, img_size)
    y = torch.randint(0, num_domains, (1,))

    disc.to(device)
    output = disc(input.to(device), y.to(device))

    assert output.shape == (1, ) 