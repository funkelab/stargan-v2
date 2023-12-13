"""
Various versions of conversion rates
"""
import argparse
from core.data_loader import get_eval_loader
from funlib.learn.torch.models import Vgg2D
import torch
from tqdm import tqdm
import numpy as np


@torch.no_grad()
def calculate_conversion_given_path(path, target_class, img_size=128, batch_size=50, num_outs_per_domain=10):
    print("Calculating conversion given path %s..." % path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classifier = Vgg2D(input_size=(128, 128), fmaps=12)
    checkpoint = torch.load("/nrs/funke/adjavond/checkpoints/synapses/classifier/vgg_checkpoint")
    classifier.load_state_dict(checkpoint["model_state_dict"])
    classifier.to(device)
    classifier.eval()

    loader = get_eval_loader(path, img_size, batch_size, imagenet_normalize=False, shuffle=False)

    predictions = []
    for x in tqdm(loader, total=len(loader)):
        x = x.to(device)
        predictions.append(
            classifier(x).cpu().numpy()
        )
    predictions = np.concatenate(predictions, axis=0)
    # Do it in a vectorized way, by reshaping the predictions
    predictions = predictions.reshape(-1, num_outs_per_domain, predictions.shape[-1])
    predictions = predictions.argmax(axis=-1)
    
    at_least_one = np.any(predictions == target_class, axis=1)
    # 
    how_many = np.sum(predictions == target_class, axis=1)
    how_many_avg = np.mean(how_many)
    conversion_rate = np.mean(at_least_one) # (sum(at_least_one) / len(at_least_one)
    translation_rate = np.mean(predictions == target_class)
    return translation_rate, conversion_rate, how_many_avg


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, help="path to images", required=True)
    parser.add_argument("--target_class", type=int, help="target class", required=True)
    parser.add_argument("--img_size", type=int, default=128, help="image size")
    parser.add_argument("--batch_size", type=int, default=50, help="batch size")
    parser.add_argument("--num_outs_per_domain", type=int, default=10, help="number of outputs per domain")
    args = parser.parse_args()
    translation_rate, conversion_rate, how_many = calculate_conversion_given_path(args.path, args.target_class, args.img_size, args.batch_size, args.num_outs_per_domain)
    print("Translation rate: %f" % translation_rate)
    print("Conversion rate: %f" % conversion_rate)
    print("How many: %f" % how_many)