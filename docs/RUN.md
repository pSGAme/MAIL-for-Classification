# Training and Evaluation

We provide bash scripts in [scripts/](../scripts) for MAIL and other methods.
Make sure to configure the dataset paths in environment variable `DATA` and 
run the commands from the main directory `MAIL-for-Classification`.
Below we provide training and evaluation instructions for MAIL. 

### Training time and compute
We train MAIL on each dataset with a batch size of 4 (64 for ImageNet and SUN397 for their relatively large scale) using a **single** NVIDIA 4090 GPU.

## MAIL

#### (1) Base-to-Novel class generalization setting
The default training settings are provided in config file at `configs/trainers/MAILSRC_Trainer/base2new.yaml`. All hyper-parameters such as prompt length, prompt depth, etc., can be modified using this config file.

Below, we provide instructions to train MAIL on imagenet. 


```bash
# Other possible dataset values includes [caltech101, food101, dtd, ucf101, oxford_flowers, oxford_pets, fgvc_aircraft, stanford_cars, sun397, eurosat]

bash scripts/mailsrc/base2new_mail.sh imagenet
```


#### (2) Cross-Dataset Transfer & Domain Generalization
We provide instructions to train MAIL on imageNet using all 1000 classes and then evaluating it directly on new downstream datasets.
We provide cross-dataset config for MAIL: `configs/MAILSRC_Trainer/cross_datasets.yaml`.
* just run (for all 3 seeds).

```bash
bash scripts/mailsrc/xd.sh
```


