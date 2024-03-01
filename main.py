"""
StarGAN v2
Copyright (c) 2020-present NAVER Corp.

This work is licensed under the Creative Commons Attribution-NonCommercial
4.0 International License. To view a copy of this license, visit
http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
"""

import os

from munch import Munch
from torch.backends import cudnn
import torch


from starganv2.options import build_parser
from starganv2.core.data_loader import get_train_loader
from starganv2.core.data_loader import get_test_loader
from starganv2.core.solver import Solver


def str2bool(v):
    return v.lower() in ("true")


def subdirs(dname):
    return [d for d in os.listdir(dname) if os.path.isdir(os.path.join(dname, d))]


def main(args):
    print(args)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)

    print("Initializing the solver...")
    solver = Solver(args)
    print("Solver initialized.")

    if args.mode == "train":
        grayscale = (args.input_dim == 1)
        print("Checking the number of domains in the training and validation directories...")
        assert len(subdirs(args.train_img_dir)) == args.num_domains
        assert len(subdirs(args.val_img_dir)) == args.num_domains
        print("Getting the training and validation loaders...")
        loaders = Munch(
            src=get_train_loader(
                root=args.train_img_dir,
                which="source",
                img_size=args.img_size,
                batch_size=args.batch_size,
                prob=args.randcrop_prob,
                num_workers=args.num_workers,
                grayscale=grayscale,
            ),
            ref=get_train_loader(
                root=args.train_img_dir,
                which="reference",
                img_size=args.img_size,
                batch_size=args.batch_size,
                prob=args.randcrop_prob,
                num_workers=args.num_workers,
                grayscale=grayscale,
            ),
            val=get_test_loader(
                root=args.val_img_dir,
                img_size=args.img_size,
                batch_size=args.val_batch_size,
                shuffle=True,
                num_workers=args.num_workers,
                grayscale=grayscale,
            ),
        )
        print("Training the model...")
        solver.train(loaders)
    elif args.mode == "sample":
        assert len(subdirs(args.src_dir)) == args.num_domains
        assert len(subdirs(args.ref_dir)) == args.num_domains
        loaders = Munch(
            src=get_test_loader(
                root=args.src_dir,
                img_size=args.img_size,
                batch_size=args.val_batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                grayscale=grayscale,
            ),
            ref=get_test_loader(
                root=args.ref_dir,
                img_size=args.img_size,
                batch_size=args.val_batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                grayscale=grayscale,
            ),
        )
        solver.sample(loaders)
    elif args.mode == "eval":
        solver.evaluate()
    # elif args.mode == 'align':
    #     from core.wing import align_faces
    #     align_faces(args, args.inp_dir, args.out_dir)
    elif args.mode == "styles":
        solver.make_styles()
    else:
        raise NotImplementedError



if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    main(args)
