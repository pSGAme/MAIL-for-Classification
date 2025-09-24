TODO

# MAIL: Multi-Modal Interactive Agent Layer for Few-Shot UCDR and Beyond [NeurIPS 2025]

[//]: # ()
[//]: # ()
[//]: # ()
[//]: # (> [**MaPLe: Multi-modal Prompt Learning**]&#40;https://arxiv.org/abs/2210.03117&#41;<br>)

[//]: # (> [Muhammad Uzair Khattak]&#40;https://scholar.google.com/citations?user=M6fFL4gAAAAJ&hl=en&authuser=1&#41;, [Hanoona Rasheed]&#40;https://scholar.google.com/citations?user=yhDdEuEAAAAJ&hl=en&authuser=1&oi=sra&#41;, [Muhammad Maaz]&#40;https://scholar.google.com/citations?user=vTy9Te8AAAAJ&hl=en&authuser=1&oi=sra&#41;, [Salman Khan]&#40;https://salman-h-khan.github.io/&#41;, [Fahad Shahbaz Khan]&#40;https://scholar.google.es/citations?user=zvaeYnUAAAAJ&hl=en&#41;)

[//]: # ()
[//]: # ()
[//]: # ([![Website]&#40;https://img.shields.io/badge/Project-Website-87CEEB&#41;]&#40;https://muzairkhattak.github.io/multimodal-prompt-learning/&#41;)

[//]: # ([![paper]&#40;https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg&#41;]&#40;https://arxiv.org/abs/2210.03117&#41;)

[//]: # ([![video]&#40;https://img.shields.io/badge/Video-Presentation-F9D371&#41;]&#40;https://youtu.be/fmULeaqAzfg&#41;)

[//]: # ([![slides]&#40;https://img.shields.io/badge/Presentation-Slides-B762C1&#41;]&#40;https://drive.google.com/file/d/1GYei-3wjf4OgBVKi9tAzeif606sHBlIA/view?usp=share_link&#41;)

[//]: # ()
[//]: # ()
[//]: # (Official implementation of the paper "[MaPLe: Multi-modal Prompt Learning]&#40;https://arxiv.org/abs/2210.03117&#41;".)

[//]: # (<hr />)

[//]: # ()
[//]: # (Base-to-novel generalization:)

[//]: # ()
[//]: # ([![PWC]&#40;https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/maple-multi-modal-prompt-learning/prompt-engineering-on-imagenet&#41;]&#40;https://paperswithcode.com/sota/prompt-engineering-on-imagenet?p=maple-multi-modal-prompt-learning&#41;)

[//]: # ([![PWC]&#40;https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/maple-multi-modal-prompt-learning/prompt-engineering-on-sun397&#41;]&#40;https://paperswithcode.com/sota/prompt-engineering-on-sun397?p=maple-multi-modal-prompt-learning&#41;)

[//]: # ([![PWC]&#40;https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/maple-multi-modal-prompt-learning/prompt-engineering-on-eurosat&#41;]&#40;https://paperswithcode.com/sota/prompt-engineering-on-eurosat?p=maple-multi-modal-prompt-learning&#41;)

[//]: # ([![PWC]&#40;https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/maple-multi-modal-prompt-learning/prompt-engineering-on-ucf101&#41;]&#40;https://paperswithcode.com/sota/prompt-engineering-on-ucf101?p=maple-multi-modal-prompt-learning&#41;)

[//]: # ([![PWC]&#40;https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/maple-multi-modal-prompt-learning/prompt-engineering-on-fgvc-aircraft&#41;]&#40;https://paperswithcode.com/sota/prompt-engineering-on-fgvc-aircraft?p=maple-multi-modal-prompt-learning&#41;)

[//]: # ()
[//]: # ()
[//]: # (Domain Generalization:)

[//]: # ()
[//]: # ([![PWC]&#40;https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/maple-multi-modal-prompt-learning/prompt-engineering-on-imagenet-r&#41;]&#40;https://paperswithcode.com/sota/prompt-engineering-on-imagenet-r?p=maple-multi-modal-prompt-learning&#41;)

[//]: # ([![PWC]&#40;https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/maple-multi-modal-prompt-learning/prompt-engineering-on-imagenet-a&#41;]&#40;https://paperswithcode.com/sota/prompt-engineering-on-imagenet-a?p=maple-multi-modal-prompt-learning&#41;)

[//]: # ([![PWC]&#40;https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/maple-multi-modal-prompt-learning/prompt-engineering-on-imagenet-s&#41;]&#40;https://paperswithcode.com/sota/prompt-engineering-on-imagenet-s?p=maple-multi-modal-prompt-learning&#41;)

[//]: # ()
[//]: # ()
[//]: # (<hr />)

[//]: # ()
[//]: # (# :rocket: News)

[//]: # (* **&#40;July 17, 2023&#41;**)

[//]: # (  * Our work on proposing a [Self-Regularization Framework for Prompt Learning]&#40;https://muzairkhattak.github.io/PromptSRC/&#41; has been accepted to ICCV-2023  :tada: The code is also publicly available [here]&#40;https://github.com/muzairkhattak/PromptSRC&#41;!)

[//]: # (* **&#40;Feb 28, 2023&#41;**)

[//]: # (  * Paper accepted at CVPR 2023 :tada: )

[//]: # (* **&#40;Oct 06, 2022&#41;** )

[//]: # (  * Training and evaluation codes for [MaPLe]&#40;configs/trainers/MaPLe&#41;, along with pretrained models are released.)

[//]: # (  * The repository also supports)

[//]: # ([CoOp]&#40;configs/trainers/CoOp&#41;,)

[//]: # ([Co-CoOp]&#40;configs/trainers/CoCoOp&#41;,)

[//]: # ([Deep Vision Prompting]&#40;configs/trainers/VPT/vit_b16_c2_ep5_batch4_4.yaml&#41;,)

[//]: # ([Deep Language Prompting]&#40;configs/trainers/IVLP/vit_b16_c2_ep5_batch4_4ctx_language_only.yaml&#41;, and )

[//]: # ([Independent V-L Prompting]&#40;configs/trainers/IVLP/vit_b16_c2_ep5_batch4_2+2ctx.yaml&#41;)

[//]: # (architectures.)

[//]: # (<hr />)

[//]: # ()
[//]: # (## Highlights)

[//]: # ()
[//]: # (![main figure]&#40;docs/main_figure.png&#41;)

[//]: # (> **<p align="justify"> Abstract:** *Pre-trained vision-language &#40;V-L&#41; models such as CLIP have shown excellent )

[//]: # (> generalization ability to downstream tasks. However, they are sensitive to the choice of input text prompts and )

[//]: # (> require careful selection of prompt templates to perform well. Inspired by the Natural Language Processing &#40;NLP&#41; )

[//]: # (> literature, recent CLIP adaptation approaches learn prompts as the textual inputs to fine-tune CLIP for downstream )

[//]: # (> tasks. We note that using prompting to adapt representations in a single branch of CLIP &#40;language or vision&#41; is )

[//]: # (> sub-optimal since it does not allow the flexibility to dynamically adjust both representation spaces on a downstream )

[//]: # (> task. In this work, we propose Multi-modal Prompt Learning &#40;MaPLe&#41; for both vision and language branches to improve )

[//]: # (> alignment between the vision and language representations. Our design promotes strong coupling between the )

[//]: # (> vision-language prompts to ensure mutual synergy and discourages learning independent uni-modal solutions. )

[//]: # (> Further, we learn separate prompts across different early stages to progressively model the stage-wise feature )

[//]: # (> relationships to allow rich context learning. We evaluate the effectiveness of our approach on three representative )

[//]: # (> tasks of generalization to novel classes, new target datasets and unseen domain shifts. Compared with the )

[//]: # (> state-of-the-art method Co-CoOp, MaPLe exhibits favorable performance and achieves an absolute gain of 3.45% on novel )

[//]: # (> classes and 2.72% on overall harmonic-mean, averaged over 11 diverse image recognition datasets. Our code and models )

[//]: # (> will be publicly released.* </p>)

[//]: # ()
[//]: # (## Main Contributions)

[//]: # ()
[//]: # (1&#41; **Multi-modal prompt learning:** Adapt CLIP using a novel prompting technique which prompts both the vision and language branch of CLIP.)

[//]: # (2&#41; **Vision and Language Prompt Coupling:** Explicitly condition vision prompts on their language counterparts and act as a bridge)

[//]: # (between the two modalities by allowing mutual propagation of gradients to promote synergy.)

[//]: # (3&#41; **Vision and Language Deep Prompting:** Learn multi-modal prompts across multiple transformer blocks in both vision and)

[//]: # (language branches to progressively learn the synergistic behaviour of both modalities.)

[//]: # ()
[//]: # ()
[//]: # (## :ballot_box_with_check: Supported Methods)

[//]: # ()
[//]: # ([comment]: <> &#40;| Language Prompting            | MaPLe |  [link]&#40;configs/trainers/IVLP/vit_b16_c2_ep5_batch4_4ctx_language_only.yaml&#41;      |      |&#41;)

[//]: # ()
[//]: # (| Method                    | Paper                                         |                             Configs                             |          Training Scripts          |)

[//]: # (|---------------------------|:----------------------------------------------|:---------------------------------------------------------------:|:----------------------------------:|)

[//]: # (| MaPLe                     | [CVPR 2023]&#40;https://arxiv.org/abs/2210.03117&#41;                                     | [link]&#40;configs/trainers/MaPLe/vit_b16_c2_ep5_batch4_2ctx.yaml&#41;  |       [link]&#40;scripts/maple&#41;        |)

[//]: # (| CoOp                      | [IJCV 2022]&#40;https://arxiv.org/abs/2109.01134&#41; |                  [link]&#40;configs/trainers/CoOp&#41;                  |        [link]&#40;scripts/coop&#41;        |)

[//]: # (| Co-CoOp                   | [CVPR 2022]&#40;https://arxiv.org/abs/2203.05557&#41; |                 [link]&#40;configs/trainers/CoCoOp&#41;                 |       [link]&#40;scripts/cocoop&#41;       |)

[//]: # (| Deep Vision Prompting     | -                                             |    [link]&#40;configs/trainers/VPT/vit_b16_c2_ep5_batch4_4.yaml&#41;    |        [link]&#40;scripts/vpt&#41;         |)

[//]: # (| Deep Language Prompting   | -                                             |                 [link]&#40;configs/trainers/IVLP/vit_b16_c2_ep5_batch4_4ctx_language_only.yaml&#41;                  | [link]&#40;scripts/language-prompting&#41; |)

[//]: # (| Independent V-L Prompting | -                                             | [link]&#40;configs/trainers/IVLP/vit_b16_c2_ep5_batch4_2+2ctx.yaml&#41; |  [link]&#40;scripts/independent-vlp&#41;   |)

[//]: # ()
[//]: # (<hr />)

[//]: # ()
[//]: # (## Results)

[//]: # (### MaPLe in comparison with existing methods)

[//]: # (Results reported below show accuracy for base and novel classes for across 11 recognition datasets averaged over 3 seeds.)

[//]: # ()
[//]: # (| Name                                                      | Base Acc. | Novel Acc. |    HM     | Epochs | )

[//]: # (|-----------------------------------------------------------|:---------:|:----------:|:---------:|:------:|)

[//]: # (| [CLIP]&#40;https://arxiv.org/abs/2103.00020&#41;                  |   69.34   |   74.22    |   71.70   |   -    |  )

[//]: # (| [CoOp]&#40;https://arxiv.org/abs/2109.01134&#41;                  | **82.69** |   63.22    |   71.66   |  200   | )

[//]: # (| [CoCoOp]&#40;https://arxiv.org/abs/2203.05557&#41; |   80.47   |   71.69    |   75.83   |   10   | )

[//]: # (| [MaPLe &#40;ours&#41;]&#40;https://arxiv.org/abs/2210.03117&#41;  |   82.28   | **75.14**  | **78.55** |   5    |  )

[//]: # ()
[//]: # (## Installation )

[//]: # (For installation and other package requirements, please follow the instructions detailed in [INSTALL.md]&#40;docs/INSTALL.md&#41;. )

[//]: # ()
[//]: # (## Data preparation)

[//]: # (Please follow the instructions at [DATASETS.md]&#40;docs/DATASETS.md&#41; to prepare all datasets.)

[//]: # ()
[//]: # (## Model Zoo)

[//]: # ()
[//]: # (### Vision-Language prompting methods)

[//]: # (| Name  &#40;configs&#41;                                                                                | Base Acc. | Novel Acc. |    HM     | Epochs |                                         Model / Logs                                         |)

[//]: # (|------------------------------------------------------------------------------------------------|:---------:|:----------:|:---------:|:------:|:--------------------------------------------------------------------------------------------:|)

[//]: # (| [Deep Vision Prompting]&#40;configs/trainers/VPT/vit_b16_c2_ep5_batch4_4.yaml&#41;                     |   80.24   |   73.43    |   76.68   |   5    |        [link]&#40;https://drive.google.com/drive/folders/1zJnaod8UVvo1HuxNzymLhBBS_OHq6cYp?usp=sharing&#41;                                                                                      | )

[//]: # (| [Deep Language Prompting]&#40;configs/trainers/IVLP/vit_b16_c2_ep5_batch4_4ctx_language_only.yaml&#41; |   81.72   |   73.81    |   77.56   |   5    | [link]&#40;https://drive.google.com/drive/folders/1PPLtvQIGprRUyxPiTwOSEh_oQ46zQfCN?usp=sharing&#41; |)

[//]: # (| [Independent V-L Prompting]&#40;configs/trainers/IVLP/vit_b16_c2_ep5_batch4_2+2ctx.yaml&#41;           |   82.15   |   74.07    |   77.90   |   5    | [link]&#40;https://drive.google.com/drive/folders/14NxzrRirK2GfyfWajsEGDiWa2suJoTBW?usp=sharing&#41; |)

[//]: # (| [MaPLe]&#40;configs/trainers/MaPLe/vit_b16_c2_ep5_batch4_2ctx.yaml&#41;                                | **82.28** | **75.14**  | **78.55** |   5    | [link]&#40;https://drive.google.com/drive/folders/1EvuvgR8566bL0T7ucvAL3LFVwuUPMRas?usp=sharing&#41; |)

[//]: # ()
[//]: # ()
[//]: # (## Training and Evaluation)

[//]: # (Please refer to the [RUN.md]&#40;docs/RUN.md&#41; for detailed instructions on training, evaluating and reproducing the results using our pre-trained models.)

[//]: # ()
[//]: # ()
[//]: # (<hr />)

[//]: # ()
[//]: # (## Citation)

[//]: # (If you use our work, please consider citing:)

[//]: # (```bibtex)

[//]: # (@inproceedings{khattakMaPLe,)

[//]: # (    title={MaPLe: Multi-modal Prompt Learning},)

[//]: # (    author={khattak, Muhammad Uzair and Rasheed, Hanoona and Maaz, Muhammad and Khan, Salman and Khan, Fahad Shahbaz},)

[//]: # (    booktitle={The IEEE/CVF Conference on Computer Vision and Pattern Recognition},)

[//]: # (    year={2023})

[//]: # (})

[//]: # (```)

[//]: # ()
[//]: # (## Contact)

[//]: # (If you have any questions, please create an issue on this repository or contact at uzair.khattak@mbzuai.ac.ae or hanoona.bangalath@mbzuai.ac.ae.)

[//]: # ()
[//]: # ()
[//]: # (## Acknowledgements)

[//]: # ()
[//]: # (Our code is based on [Co-CoOp and CoOp]&#40;https://github.com/KaiyangZhou/CoOp&#41; repository. We thank the authors for releasing their code. If you use our model and code, please consider citing these works as well.)

[//]: # ()
