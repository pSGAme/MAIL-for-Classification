"""
BATCH SIZE: 4 for all datasets, except for imagenet and sun397 (64) for their large size.
LR: 0.000015 for all datasets, except for food101 (0.000005) and dtd (0.000025).
RANK: 32 for all datasets, except for dtd (12).
RRCROP_SCALE: (0.5, 1.0) for all datasets, except for stanford_cars (0.45, 1.0).
Optimizer: AdamW for all datasets, except for EuroSAT, we find that SGD for EuroSAT is much better.
"""
def get_dataset_specified_config(dataset):
    """Get dataset specific."""
    cfg = {
        "ImageNet": {
            "OPTIM.MAX_EPOCH": 5,
            "DATALOADER.TRAIN_X.BATCH_SIZE": 64,
        },
        "SUN397": {
            "DATALOADER.TRAIN_X.BATCH_SIZE": 64,
        },
        "OxfordPets": {
            "TRAINER.MAILSRC_Trainer.PROJ": False,
        },
        "StanfordCars": {
            "INPUT.RRCROP_SCALE": (0.45, 1.0),
        },
        "Food101": {
            "OPTIM.LR": 0.000005,
        },
        "DescribableTextures": {
            "OPTIM.LR": 0.000025,
            "TRAINER.MAILSRC_Trainer.RANK": 12,
        },
        "EuroSAT": {
            "OPTIM.NAME": "sgd",
        },
        "Caltech101": {
            "INPUT.TRANSFORMS": ["randaugment", "normalize"]
        },
    }.get(dataset, {})

    return [item for pair in cfg.items() for item in pair]