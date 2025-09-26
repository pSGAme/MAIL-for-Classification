import argparse
import os
import sys

import torch
from dassl.utils import setup_logger, set_random_seed, collect_env_info
from dassl.config import get_cfg_default
from dassl.engine import build_trainer

# multi card
# import
import torch.multiprocessing as mp


# custom
import datasets.oxford_pets
import datasets.oxford_flowers
import datasets.fgvc_aircraft
import datasets.dtd
import datasets.eurosat
import datasets.stanford_cars
import datasets.food101
import datasets.sun397
import datasets.caltech101
import datasets.ucf101
import datasets.imagenet

import datasets.imagenet_sketch
import datasets.imagenetv2
import datasets.imagenet_a
import datasets.imagenet_r

import trainers.coop
import trainers.cocoop
import trainers.zsclip
import trainers.maple
import trainers.independentVL
import trainers.vpt
from trainers.config import get_dataset_specified_config
import trainers.mailsrc


def get_prompt(cfg_filname: str) -> tuple:
    pre = "a photo of a"
    post = "."
    if cfg_filname.endswith("caltech101.yaml"):
        pre = "a drawing of a"
    elif cfg_filname.endswith("oxford_pets.yaml"):
        pre = "an awesome animal pet photo of a"
    elif cfg_filname.endswith("stanford_cars.yaml"):
        pre = "a photo of my"  # cross
    elif cfg_filname.endswith("oxford_flowers.yaml"):
        pre = "a flower photo of a"
        # pre = "a beautiful flower photo of a" # cross`Cc
    elif cfg_filname.endswith("food101.yaml"):
        pre = "a yummy food photo of a"
    elif cfg_filname.endswith("fgvc_aircraft.yaml"):
        pre = "a brand aircraft of a"
        #pre = "an awesome brand aircraft of my"  # cross
    elif cfg_filname.endswith("sun397.yaml"):
        pre = "a scene photo of a"  # cross
    elif cfg_filname.endswith("dtd.yaml"):
        # pre = "a texture photo of a"
        pre = "a beautiful texture drawing of a"
        #post = "texture."  # cross
    elif cfg_filname.endswith("eurosat.yaml"):
        # pre = "an image of a"
        # post = ", a type of very centered satellite"
        pre = "a photo of a"
        post = ", a type of centered satellite"
    elif cfg_filname.endswith("ucf101.yaml"):
        post = "a type of action."
    elif cfg_filname.endswith("imagenet_sketch.yaml"):
        pre = "a sketch photo of a"
    elif cfg_filname.endswith("imagenet_a.yaml"):
        pre = "a poor photo of a"
    elif cfg_filname.endswith("imagenet_r.yaml"):
        pre = "a sketch photo of a"
    return pre, post


def print_args(args, cfg):
    print("***************")
    print("** Arguments **")
    print("***************")
    optkeys = list(args.__dict__.keys())
    optkeys.sort()
    for key in optkeys:
        print("{}: {}".format(key, args.__dict__[key]))
    print("************")
    print("** Config **")
    print("************")
    print(cfg)


def reset_cfg(cfg, args):
    if args.root:
        cfg.DATASET.ROOT = args.root

    if args.output_dir:
        cfg.OUTPUT_DIR = args.output_dir

    if args.resume:
        cfg.RESUME = args.resume

    if args.seed:
        cfg.SEED = args.seed

    if args.source_domains:
        cfg.DATASET.SOURCE_DOMAINS = args.source_domains

    if args.target_domains:
        cfg.DATASET.TARGET_DOMAINS = args.target_domains

    if args.transforms:
        cfg.INPUT.TRANSFORMS = args.transforms

    if args.trainer:
        cfg.TRAINER.NAME = args.trainer

    if args.backbone:
        cfg.MODEL.BACKBONE.NAME = args.backbone

    if args.head:
        cfg.MODEL.HEAD.NAME = args.head


def extend_cfg(cfg, args):
    """
    Add new config variables.

    E.g.
        from yacs.config import CfgNode as CN
        cfg.TRAINER.MY_MODEL = CN()
        cfg.TRAINER.MY_MODEL.PARAM_A = 1.
        cfg.TRAINER.MY_MODEL.PARAM_B = 0.5
        cfg.TRAINER.MY_MODEL.PARAM_C = False
    """
    from yacs.config import CfgNode as CN

    cfg.TRAINER.COOP = CN()
    cfg.TRAINER.COOP.N_CTX = 16  # number of context vectors
    cfg.TRAINER.COOP.CSC = False  # class-specific context
    cfg.TRAINER.COOP.CTX_INIT = ""  # initialization words
    cfg.TRAINER.COOP.PREC = "fp16"  # fp16, fp32, amp
    cfg.TRAINER.COOP.CLASS_TOKEN_POSITION = "end"  # 'middle' or 'end' or 'front'

    cfg.TRAINER.COCOOP = CN()
    cfg.TRAINER.COCOOP.N_CTX = 16  # number of context vectors
    cfg.TRAINER.COCOOP.CTX_INIT = ""  # initialization words
    cfg.TRAINER.COCOOP.PREC = "fp16"  # fp16, fp32, amp

    # Config for MAIL
    cfg.TRAINER.MAILSRC_Trainer = CN()
    cfg.TRAINER.MAILSRC_Trainer.RANK = 32  # rank of the bridge function
    cfg.TRAINER.MAILSRC_Trainer.IVLU_START_LAYER = 1  # start layer of ivlu
    cfg.TRAINER.MAILSRC_Trainer.IVLU_END_LAYER = 12  # end layer of ivlu
    cfg.TRAINER.MAILSRC_Trainer.START_LAYER = 1
    cfg.TRAINER.MAILSRC_Trainer.END_LAYER = 12  # start and the end layers of applying MAIL
    prefix, postfix = get_prompt(args.dataset_config_file)
    cfg.TRAINER.MAILSRC_Trainer.PREFIX_INIT = prefix  # initialization words
    cfg.TRAINER.MAILSRC_Trainer.POSTFIX_INIT = postfix  # initialization words
    cfg.TRAINER.MAILSRC_Trainer.PREC = "fp16"  # fp16, fp32, amp
    cfg.TRAINER.MAILSRC_Trainer.TEXT_LOSS_WEIGHT = 0.0
    cfg.TRAINER.MAILSRC_Trainer.IMAGE_LOSS_WEIGHT = 0.0
    cfg.TRAINER.MAILSRC_Trainer.LOGIT_LOSS_WEIGHT = 0.0
    cfg.TRAINER.MAILSRC_Trainer.PROJ = True
    cfg.DATASET.SUBSAMPLE_CLASSES = "all"  # all, base or new


    # Config for MaPLe
    cfg.TRAINER.MAPLE = CN()
    cfg.TRAINER.MAPLE.N_CTX = 2  # number of context vectors
    cfg.TRAINER.MAPLE.CTX_INIT = "a photo of a"  # initialization words
    cfg.TRAINER.MAPLE.PREC = "fp16"  # fp16, fp32, amp
    cfg.TRAINER.MAPLE.PROMPT_DEPTH = 9  # Max 12, minimum 0, for 1 it will act as shallow MaPLe (J=1)
    cfg.DATASET.SUBSAMPLE_CLASSES = "all"  # all, base or new

    # Config for independent Vision Language prompting (independent-vlp)
    cfg.TRAINER.IVLP = CN()
    cfg.TRAINER.IVLP.N_CTX_VISION = 2  # number of context vectors at the vision branch
    cfg.TRAINER.IVLP.N_CTX_TEXT = 2  # number of context vectors at the language branch
    cfg.TRAINER.IVLP.CTX_INIT = "a photo of a"  # initialization words (only for language prompts)
    cfg.TRAINER.IVLP.PREC = "fp16"  # fp16, fp32, amp
    # If both variables below are set to 0, 0, will the config will degenerate to COOP model
    cfg.TRAINER.IVLP.PROMPT_DEPTH_VISION = 9  # Max 12, minimum 0, for 0 it will act as shallow MaPLe (J=1)
    cfg.TRAINER.IVLP.PROMPT_DEPTH_TEXT = 9  # Max 12, minimum 0, for 0 it will act as shallow MaPLe (J=1)
    cfg.DATASET.SUBSAMPLE_CLASSES = "all"  # all, base or new

    # Config for only vision side prompting
    cfg.TRAINER.VPT = CN()
    cfg.TRAINER.VPT.N_CTX_VISION = 2  # number of context vectors at the vision branch
    cfg.TRAINER.VPT.CTX_INIT = "a photo of a"  # initialization words
    cfg.TRAINER.VPT.PREC = "fp16"  # fp16, fp32, amp
    cfg.TRAINER.VPT.PROMPT_DEPTH_VISION = 1  # if set to 1, will represent shallow vision prompting only
    cfg.DATASET.SUBSAMPLE_CLASSES = "all"  # all, base or new


def setup_cfg(args):
    cfg = get_cfg_default()
    extend_cfg(cfg, args)

    # 1. From the dataset config file
    if args.dataset_config_file:
        cfg.merge_from_file(args.dataset_config_file)

    # 2. From the method config file
    if args.config_file:
        cfg.merge_from_file(args.config_file)

    # 3. From input arguments
    reset_cfg(cfg, args)


    # 4. From optional input arguments
    cfg.merge_from_list(args.opts)

    # 5. Override dataset specific config
    if cfg.DATASET.SUBSAMPLE_CLASSES != "all":  # FOR BASE2NEW
        cfg.merge_from_list(get_dataset_specified_config(cfg.DATASET.NAME))

    cfg.freeze()

    return cfg


def main(args):
    cfg = setup_cfg(args)
    if cfg.SEED >= 0:
        print("Setting fixed seed: {}".format(cfg.SEED))
        set_random_seed(cfg.SEED)
    setup_logger(cfg.OUTPUT_DIR)

    if torch.cuda.is_available() and cfg.USE_CUDA:
        torch.backends.cudnn.benchmark = True

    print_args(args, cfg)
    print("Collecting env info ...")
    print("** System info **\n{}\n".format(collect_env_info()))

    trainer = build_trainer(cfg)

    if args.eval_only:
        trainer.load_model(args.model_dir, epoch=args.load_epoch)
        trainer.test()
        # trainer.test_cka()
        return

    if not args.no_train:
        trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="", help="path to dataset")
    parser.add_argument("--output-dir", type=str, default="", help="output directory")
    parser.add_argument(
        "--resume",
        type=str,
        default="",
        help="checkpoint directory (from which the training resumes)",
    )
    parser.add_argument(
        "--seed", type=int, default=-1, help="only positive value enables a fixed seed"
    )
    parser.add_argument(
        "--source-domains", type=str, nargs="+", help="source domains for DA/DG"
    )
    parser.add_argument(
        "--target-domains", type=str, nargs="+", help="target domains for DA/DG"
    )
    parser.add_argument(
        "--transforms", type=str, nargs="+", help="data augmentation methods"
    )
    parser.add_argument(
        "--config-file", type=str, default="", help="path to config file"
    )
    parser.add_argument(
        "--dataset-config-file",
        type=str,
        default="",
        help="path to config file for dataset setup",
    )
    parser.add_argument("--trainer", type=str, default="", help="name of trainer")
    parser.add_argument("--backbone", type=str, default="", help="name of CNN backbone")
    parser.add_argument("--head", type=str, default="", help="name of head")
    parser.add_argument("--eval-only", action="store_true", help="evaluation only")
    parser.add_argument(
        "--model-dir",
        type=str,
        default="",
        help="load model from this directory for eval-only mode",
    )
    parser.add_argument(
        "--load-epoch", type=int, help="load model weights at this epoch for evaluation"
    )
    parser.add_argument(
        "--no-train", action="store_true", help="do not call trainer.train()"
    )
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="modify config options using the command-line",
    )
    args = parser.parse_args()
    main(args)
    sys.stdout.flush()
    sys.stderr.flush()
    print("finished!!!!!!!!!!!!!!!!!!!!")
    os._exit(0)
