from starganv2.core.data_loader import get_eval_loader
from starganv2.core.utils import denormalize
from funlib.learn.torch.models import Vgg2D
import numpy as np
import os
import torch
from tqdm import tqdm
import zarr


@torch.no_grad()
def generate_styles(nets, args, step):
    print("Generating styles...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    domains = os.listdir(args.val_img_dir)
    domains.sort()
    offset = 0
    domains = domains[offset:]
    num_domains = len(domains)
    print("Number of domains: %d" % num_domains)

    output = zarr.open(os.path.join(args.eval_dir, "results_%d.zarr" % step), "a")

    output_styles = output.require_group("styles")
    output_images = output.require_group("images")
    output_predictions = output.require_group("predictions")

    # Load the classifier
    classifier = Vgg2D(input_size=(128, 128), fmaps=12)
    checkpoint = torch.load("/nrs/funke/adjavond/checkpoints/synapses/classifier/vgg_checkpoint")
    classifier.load_state_dict(checkpoint["model_state_dict"])
    classifier.to(device)
    classifier.eval()

    for trg_idx, domain in enumerate(domains): 
        # Generate images and predictions
        path_ref = os.path.join(args.val_img_dir, domain)
        loader_ref = get_eval_loader(
            root=path_ref,
            img_size=args.img_size,
            batch_size=args.val_batch_size,
            imagenet_normalize=False,
            shuffle=False,
            drop_last=False
        )
        imgs = []
        preds = []
        styles = []
        for x in tqdm(loader_ref, total=len(loader_ref)):
            imgs.append(
                (denormalize(x) * 255).cpu().numpy().astype(np.uint8)
            ) 
            preds.append(
                torch.softmax(classifier(x.to(device)), dim=-1).cpu().numpy()
            )
            # Generate styles
            N = x.size(0)
            y_trg = torch.tensor([trg_idx + offset] * N).to(device)
            styles.append(
                nets.style_encoder(x.to(device), y_trg).cpu().numpy()
            )

        output_images[domain] = np.concatenate(imgs, axis=0)
        output_predictions[domain] = np.concatenate(preds, axis=0)
        output_styles[domain] = np.concatenate(styles, axis=0)


    for trg_idx, trg_domain in enumerate(domains):
        src_domains = [x for x in domains if x != trg_domain]

        path_ref = os.path.join(args.val_img_dir, trg_domain)
        loader_ref = get_eval_loader(
            root=path_ref,
            img_size=args.img_size,
            batch_size=args.val_batch_size,
            imagenet_normalize=False,
            shuffle=False,
            drop_last=False
        )

        # Start by generating styles for the reference domain
        styles = torch.from_numpy(output_styles[trg_domain][:])

        for src_idx, src_domain in enumerate(src_domains):
            print("Generating images for %s to %s..." % (src_domain, trg_domain))
            path_src = os.path.join(args.val_img_dir, src_domain)
            all_generated, all_predictions = generate_images_given_styles_and_path(
                nets, classifier, path_src, styles, args.img_size, args.val_batch_size
            )
            output_images["%s_to_%s" % (src_domain, trg_domain)] = all_generated
            output_predictions["%s_to_%s" % (src_domain, trg_domain)] = all_predictions


@torch.no_grad()
def generate_images_given_styles_and_path(
    nets, classifier, path, styles, img_size=128, batch_size=50, max_styles=500,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loader = get_eval_loader(
        path,
        img_size,
        batch_size,
        imagenet_normalize=False,
        shuffle=False,
        drop_last=False,
    )
    # Crop styles to max_styles
    styles = styles[:max_styles]
    all_generated = []
    all_predictions = []
    i = 0
    for x in tqdm(loader, total=len(loader), desc="Sources"):
        this_batch = []
        this_prediction = []
        x = x.to(device)
        N = x.size(0)
        # Create loader for style
        repeated_styles = torch.repeat_interleave(styles, N, dim=0)
        assert all(repeated_styles[0] == repeated_styles[N-1])

        style_loader = torch.utils.data.DataLoader(
            repeated_styles, batch_size=N, shuffle=False, drop_last=False
        )

        for s_trg in tqdm(style_loader, total=len(style_loader), desc="Styles"):
            x_fake = nets.generator(x, s_trg)
            this_batch.append(
                (denormalize(x_fake).cpu().numpy() * 255).astype(np.uint8)
            )
            # TODO is this fair, or should i be de-normalizing (+ clipping) and then re-normalizing?
            pred = classifier(x_fake)
            this_prediction.append(
                torch.softmax(pred, dim=-1).cpu().numpy()
            )
        this_batch = np.stack(this_batch, axis=1)
        this_prediction = np.stack(this_prediction, axis=1)
        all_generated.append(this_batch)
        all_predictions.append(this_prediction)
    all_generated = np.concatenate(all_generated, axis=0)
    all_predictions = np.concatenate(all_predictions, axis=0)

    return all_generated, all_predictions
