import argparse


def build_parser():
    parser = argparse.ArgumentParser()

    # model arguments
    parser.add_argument("--img_size", type=int, default=256, help="Image resolution")
    parser.add_argument("--num_domains", type=int, default=2, help="Number of domains")
    parser.add_argument(
        "--latent_dim", type=int, default=16, help="Latent vector dimension"
    )
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=512,
        help="Hidden dimension of mapping network",
    )
    parser.add_argument(
        "--style_dim", type=int, default=64, help="Style code dimension"
    )
    parser.add_argument(
        "--input_dim", type=int, default=3, help="Number of channels in the image"
    )
    parser.add_argument(
        "--single_output_style_encoder", action="store_true", help="Use a single output style encoder"
    )

    # weight for objective functions
    parser.add_argument(
        "--lambda_reg", type=float, default=1, help="Weight for R1 regularization"
    )
    parser.add_argument(
        "--lambda_cyc", type=float, default=1, help="Weight for cyclic consistency loss"
    )
    parser.add_argument(
        "--lambda_sty",
        type=float,
        default=1,
        help="Weight for style reconstruction loss",
    )
    parser.add_argument(
        "--lambda_ds", type=float, default=1, help="Weight for diversity sensitive loss"
    )
    parser.add_argument(
        "--ds_iter",
        type=int,
        default=100000,
        help="Number of iterations to optimize diversity sensitive loss",
    )
    parser.add_argument(
        "--w_hpf", type=float, default=0, help="weight for high-pass filtering"
    )

    # training arguments
    parser.add_argument(
        "--randcrop_prob",
        type=float,
        default=0.5,
        help="Probabilty of using random-resized cropping",
    )
    parser.add_argument(
        "--total_iters", type=int, default=100000, help="Number of total iterations"
    )
    parser.add_argument(
        "--resume_iter",
        type=int,
        default=0,
        help="Iterations to resume training/testing",
    )
    parser.add_argument(
        "--batch_size", type=int, default=8, help="Batch size for training"
    )
    parser.add_argument(
        "--val_batch_size", type=int, default=32, help="Batch size for validation"
    )
    parser.add_argument(
        "--lr", type=float, default=1e-4, help="Learning rate for D, E and G"
    )
    parser.add_argument("--f_lr", type=float, default=1e-6, help="Learning rate for F")
    parser.add_argument(
        "--beta1", type=float, default=0.0, help="Decay rate for 1st moment of Adam"
    )
    parser.add_argument(
        "--beta2", type=float, default=0.99, help="Decay rate for 2nd moment of Adam"
    )
    parser.add_argument(
        "--weight_decay", type=float, default=1e-4, help="Weight decay for optimizer"
    )
    parser.add_argument(
        "--num_outs_per_domain",
        type=int,
        default=10,
        help="Number of generated images per domain during sampling",
    )

    # misc
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["train", "sample", "eval", "align", "styles"],
        help="This argument is used in solver",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of workers used in DataLoader",
    )
    parser.add_argument(
        "--seed", type=int, default=777, help="Seed for random number generator"
    )

    # directory for training
    parser.add_argument(
        "--train_img_dir",
        type=str,
        default="data/celeba_hq/train",
        help="Directory containing training images",
    )
    parser.add_argument(
        "--val_img_dir",
        type=str,
        default="data/celeba_hq/val",
        help="Directory containing validation images",
    )
    parser.add_argument(
        "--sample_dir",
        type=str,
        default="expr/samples",
        help="Directory for saving generated images",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="expr/checkpoints",
        help="Directory for saving network checkpoints",
    )

    # directory for calculating metrics
    parser.add_argument(
        "--eval_dir",
        type=str,
        default="expr/eval",
        help="Directory for saving metrics, i.e., FID and LPIPS",
    )

    # directory for testing
    parser.add_argument(
        "--result_dir",
        type=str,
        default="expr/results",
        help="Directory for saving generated images and videos",
    )
    parser.add_argument(
        "--src_dir",
        type=str,
        default="assets/representative/celeba_hq/src",
        help="Directory containing input source images",
    )
    parser.add_argument(
        "--ref_dir",
        type=str,
        default="assets/representative/celeba_hq/ref",
        help="Directory containing input reference images",
    )
    parser.add_argument(
        "--inp_dir",
        type=str,
        default="assets/representative/custom/female",
        help="input directory when aligning faces",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="assets/representative/celeba_hq/src/female",
        help="output directory when aligning faces",
    )

    # face alignment
    parser.add_argument("--wing_path", type=str, default="expr/checkpoints/wing.ckpt")
    parser.add_argument(
        "--lm_path", type=str, default="expr/checkpoints/celeba_lm_mean.npz"
    )

    # step size
    parser.add_argument("--print_every", type=int, default=10)
    parser.add_argument("--sample_every", type=int, default=5000)
    parser.add_argument("--save_every", type=int, default=10000)
    parser.add_argument("--eval_every", type=int, default=25000)

    # Parameters for the classifier-based evaluations
    parser.add_argument("--classifier_checkpoint", type=str, help="path to the torchscript checkpoint of the classifier")
    # Data preparation arguments for the classsifier
    parser.add_argument(
        "--mean",
        type=float,
        nargs="+",
        default=0.5,
        help="The mean for data normalization. Should be either one float, or as many as there are dimensions.",
    )
    parser.add_argument(
        "--std",
        type=float,
        nargs="+",
        default=0.5,
        help="The std for data normalization. Should be the same size as mean.",
    )

    # Device setup
    parser.add_argument("--gpu_ids", type=int, nargs="+", default=[0], help="Device to use") 
    return parser