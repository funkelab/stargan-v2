"""
Evaluation metrics to check whether the images output by StarGAN v2 are correctly classified by the classifier.
"""
import argparse
from core.data_loader import get_eval_loader
from funlib.learn.torch.models import Vgg2D
import torch
from tqdm import tqdm
import numpy as np


@torch.no_grad()
def calculate_conversion_given_path(path, target_class, img_size=128, batch_size=50):
    print("Calculating conversion given path %s..." % path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classifier = Vgg2D(input_size=(128, 128), fmaps=12)
    checkpoint = torch.load("/nrs/funke/adjavond/checkpoints/synapses/classifier/vgg_checkpoint")
    classifier.load_state_dict(checkpoint["model_state_dict"])
    classifier.to(device)
    classifier.eval()

    loader = get_eval_loader(path, img_size, batch_size, imagenet_normalize=False)

    predicted_classes = []
    for x in tqdm(loader, total=len(loader)):
        x = x.to(device)
        predicted_classes.append(
            torch.argmax(classifier(x), dim=1).cpu().numpy()
        )
    predicted_classes = np.concatenate(predicted_classes, axis=0)
    conversion_rate = np.mean(predicted_classes == target_class)
    print("Conversion rate: %f" % conversion_rate)
    return conversion_rate


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, help="path to images", required=True)
    parser.add_argument("--target_class", type=int, help="target class", required=True)
    parser.add_argument("--img_size", type=int, default=128, help="image size")
    parser.add_argument("--batch_size", type=int, default=50, help="batch size")
    args = parser.parse_args()
    conversion_rate = calculate_conversion_given_path(args.path, args.target_class, args.img_size, args.batch_size)
    print("Conversion rate: %f" % conversion_rate)