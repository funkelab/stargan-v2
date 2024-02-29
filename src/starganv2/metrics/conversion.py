"""
Various versions of conversion rates
"""

import argparse
from starganv2.core.data_loader import get_eval_loader
from starganv2.core.classification import ClassifierWrapper
from funlib.learn.torch.models import Vgg2D
import torch
from torchvision import transforms
from tqdm import tqdm
import numpy as np


@torch.no_grad()
def calculate_conversion_given_path(
    path,
    model_checkpoint,
    target_class,
    img_size=128,
    batch_size=50,
    num_outs_per_domain=10,
    mean=0.5,
    std=0.5,
):
    print("Calculating conversion given path %s..." % path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classifier = ClassifierWrapper(model_checkpoint, mean=mean, std=std)
    classifier.to(device)
    classifier.eval()

    loader = get_eval_loader(
        path, img_size, batch_size, imagenet_normalize=False, shuffle=False
    )

    predictions = []
    for x in tqdm(loader, total=len(loader)):
        x = x.to(device)
        predictions.append(classifier(x).cpu().numpy())
    predictions = np.concatenate(predictions, axis=0)
    # Do it in a vectorized way, by reshaping the predictions
    predictions = predictions.reshape(-1, num_outs_per_domain, predictions.shape[-1])
    predictions = predictions.argmax(axis=-1)
    # 
    at_least_one = np.any(predictions == target_class, axis=1)
    #
    conversion_rate = np.mean(at_least_one)  # (sum(at_least_one) / len(at_least_one)
    translation_rate = np.mean(predictions == target_class)
    return translation_rate, conversion_rate


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, help="path to images", required=True)
    parser.add_argument("--model_checkpoint", type=str, help="path to model checkpoint", required=True)
    parser.add_argument("--target_class", type=int, help="target class", required=True)
    parser.add_argument("--img_size", type=int, default=128, help="image size")
    parser.add_argument("--batch_size", type=int, default=50, help="batch size")
    parser.add_argument(
        "--num_outs_per_domain",
        type=int,
        default=10,
        help="number of outputs per domain",
    )
    parser.add_argument("--mean", type=float, nargs="+", default=0.5, help="mean for normalization")
    parser.add_argument("--std", type=float, nargs="+", default=0.5, help="std for normalization")
    args = parser.parse_args()
    translation_rate, conversion_rate = calculate_conversion_given_path(
        args.path,
        model_checkpoint=args.model_checkpoint,
        target_class=args.target_class,
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_outs_per_domain=args.num_outs_per_domain,
        mean=args.mean,
        std=args.std,
    )
    print("Translation rate: %f" % translation_rate)
    print("Conversion rate: %f" % conversion_rate)
