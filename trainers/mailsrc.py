import copy
import torch
import numpy as np
import torch.nn as nn
import os.path as osp

from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast
from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler
from clip_mail import clip
from clip import clip as clip_zs
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from .imagenet_templates import IMAGENET_TEMPLATES

_tokenizer = _Tokenizer()


def load_clip_to_cpu(cfg, zero_shot_model=False):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")
    if not zero_shot_model:
        design_details = {"trainer": 'MaPLe',
                          "vision_depth": 0,
                          "language_depth": 0, "vision_ctx": 0,
                          "language_ctx": 0,
                          "maple_length": cfg.TRAINER.MAPLE.N_CTX}

        model = clip.build_model(state_dict or model.state_dict(), design_details)
    else:
        design_details = {"trainer": 'IVLP',
                          "vision_depth": 0,
                          "language_depth": 0, "vision_ctx": 0,
                          "language_ctx": 0}
        model = clip_zs.build_model(state_dict or model.state_dict(), design_details)

    return model


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

#
class MAIL(nn.Module):
    def __init__(self, cfg, visual_dim=768, text_dim=512, dtype='float16'):
        super().__init__()
        self.cfg = cfg
        self.visual_a = torch.nn.Parameter(torch.ones(visual_dim))
        self.visual_b = torch.nn.Parameter(torch.zeros(visual_dim))
        # #
        self.text_a = torch.nn.Parameter(torch.ones(text_dim))
        self.text_b = torch.nn.Parameter(torch.zeros(text_dim))
        self.dtype = cfg.TRAINER.MAILSRC_Trainer.PREC

        rank = self.cfg.TRAINER.MAILSRC_Trainer.RANK
        visual_scale = visual_dim ** -0.5
        text_scale = text_dim ** -0.5
        #
        # gaussian - 0 distribution
        self.text_proj_down = nn.Parameter(text_scale * 1 * torch.randn(text_dim, rank))
        self.text_proj_up = nn.Parameter(visual_scale * 0 * torch.randn(rank, visual_dim))

    def forward(self, x, is_text, i=0):
        if self.cfg.TRAINER.MAILSRC_Trainer.IVLU_START_LAYER <= i <= self.cfg.TRAINER.MAILSRC_Trainer.IVLU_END_LAYER:
            if is_text:
                x = self.text_forward(x, i)
            else:
                x = self.visual_forward(x, i)
        if self.dtype == "fp16":
            return x.half()
        return x

    def visual_forward(self, x, i):
        if self.cfg.TRAINER.MAILSRC_Trainer.START_LAYER <= i <= self.cfg.TRAINER.MAILSRC_Trainer.END_LAYER:
            a = self.visual_a + self.text_a @ self.text_proj_down @ self.text_proj_up
        else:
            a = self.visual_a
        b = self.visual_b
        x = x * a + b
        return x

    def text_forward(self, x, i):
        a = self.text_a
        b = self.text_b
        x = x * a + b
        return x



class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts, ln_proj_layers_mlp, ln_proj_layers_att, mlp_proj_layers_mlp,
                att_proj_layers_att, last_ln=None):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        combined = [x, 0, ln_proj_layers_mlp, ln_proj_layers_att, mlp_proj_layers_mlp, att_proj_layers_att]
        outputs = self.transformer(combined)
        x = outputs[0]  # extract the x back from here
        x = x.permute(1, 0, 2)  # LND -> NLD
        if last_ln is not None:
            x = last_ln(self.ln_final(x).type(self.dtype), is_text=True, i=12)
        else:
            x = self.ln_final(x).type(self.dtype)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


class MultiModalMailLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        prefix_init = cfg.TRAINER.MAILSRC_Trainer.PREFIX_INIT
        postfix_init = cfg.TRAINER.MAILSRC_Trainer.POSTFIX_INIT
        dtype = clip_model.dtype  # fp32

        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        prefix_init = prefix_init.replace("_", " ")
        postfix_init = postfix_init.replace("_", " ")
        prompt_prefix = prefix_init
        prompt_postfix = postfix_init

        print(f'Initial prefix context: "{prompt_prefix}"')
        print(f'Initial postfix context: "{prompt_postfix}"')

        ln_single_layer = MAIL(cfg=cfg, visual_dim=768, text_dim=512, dtype=dtype)
        ivlu_end_layer = 12
        self.ln_proj_layers_mlp = _get_clones(ln_single_layer, ivlu_end_layer)
        self.ln_proj_layers_att = _get_clones(ln_single_layer, ivlu_end_layer)
        self.att_proj_layers_att = _get_clones(ln_single_layer, ivlu_end_layer)
        self.mlp_proj_layers_mlp = _get_clones(ln_single_layer, ivlu_end_layer)
        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]

        if len(prompt_prefix) == 0:
            prompt_postfix = " " + prompt_postfix
        else:
            prompt_prefix = prompt_prefix + " "
            if len(prompt_postfix) > 1:  # greater than "."
                prompt_postfix = ", " + prompt_postfix

        prompts = [prompt_prefix + name + prompt_postfix for name in classnames]
        print("prompt example: ", prompts[0])

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])  # (n_cls, n_tkn)
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        self.register_buffer("embedding", embedding)  # CLS, EOS

        clip_model_temp = load_clip_to_cpu(cfg, True).float().cuda()
        clip_model_temp_image = load_clip_to_cpu(cfg, True)
        if cfg.TRAINER.MAILSRC_Trainer.PREC == "fp32" or cfg.TRAINER.MAILSRC_Trainer.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model_temp_image.float()

        with torch.no_grad():
            self.ZS_image_encoder = clip_model_temp_image.visual
            # Now pre-compute the frozen VL embeddings
            all_teacher_features = []
            # Using multiple text templates to ensure textual diversity during training
            # for single_template in IMAGENET_TEMPLATES:
            #     x = [single_template.replace("{}", name) for name in classnames]
            #     x_tokenized = torch.cat([clip.tokenize(p) for p in x])
            #     text_features = clip_model_temp.encode_text(x_tokenized.cuda())
            #     all_teacher_features.append(text_features.unsqueeze(1))
            text_features = clip_model_temp.encode_text(tokenized_prompts.cuda())
            all_teacher_features.append(text_features.unsqueeze(1))
        self.fixed_embeddings = torch.cat(all_teacher_features, dim=1).mean(dim=1)

        self.n_cls = n_cls
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens

    def forward(self):

        prompts = self.embedding
        return prompts, self.ln_proj_layers_mlp, self.ln_proj_layers_att, self.att_proj_layers_att, self.mlp_proj_layers_mlp
        # return prompts, None, self.compound_prompts_text, visual_deep_prompts, \
        #        self.ln_proj_layers_mlp, self.ln_proj_layers_att, self.att_proj_layers_att, self.mlp_proj_layers_mlp


class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.mail_learner = MultiModalMailLearner(cfg, classnames, clip_model)

        self.tokenized_prompts = self.mail_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)

        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.last_ln = MAIL(cfg=cfg, visual_dim=768, text_dim=512, dtype=self.dtype)
        self.last_adapter = MAIL(cfg=cfg, visual_dim=512, text_dim=512, dtype=self.dtype)
        self.cfg = cfg


    def forward(self, image, label=None):
        tokenized_prompts = self.tokenized_prompts
        logit_scale = self.logit_scale.exp()

        prompts, ln_proj_layers_mlp, \
        ln_proj_layers_att, mlp_proj_layers_mlp, att_proj_layers_att = self.mail_learner()  # i write dao zheli le
        text_features = self.text_encoder(prompts, tokenized_prompts, ln_proj_layers_mlp,
                                          ln_proj_layers_att, mlp_proj_layers_mlp, att_proj_layers_att, self.last_ln)
        image_features = self.image_encoder(image.type(self.dtype),
                                            ln_proj_layers_mlp,
                                            ln_proj_layers_att, mlp_proj_layers_mlp, att_proj_layers_att, self.last_ln)
        if self.last_adapter:
            image_features = self.last_adapter(image_features, is_text=False, i=12)
            text_features = self.last_adapter(text_features, is_text=True, i=12)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logits = logit_scale * image_features @ text_features.t()

        if self.mail_learner.training:
            # Now calculate the frozen pre-trained features
            fixed_embeddings = self.mail_learner.fixed_embeddings  # precomputed pre-trained frozen textual features
            fixed_embeddings = fixed_embeddings / fixed_embeddings.norm(dim=-1, keepdim=True)
            with torch.no_grad():
                zero_shot_features = self.mail_learner.ZS_image_encoder(image.type(self.dtype))
                # print(next(self.mail_learner.ZS_image_encoder.parameters()).device)
                zero_shot_features = zero_shot_features / zero_shot_features.norm(dim=-1, keepdim=True)
                # Compute pre-trained frozen visual features

                if self.cfg.TRAINER.MAILSRC_Trainer.PREC=="fp16":
                    zero_shot_logits = logit_scale * zero_shot_features.cuda() @ fixed_embeddings.half().cuda().t()
                else:
                    zero_shot_logits = logit_scale * zero_shot_features.cuda() @ fixed_embeddings.cuda().t()

            return F.cross_entropy(logits,
                                   label), text_features, fixed_embeddings, zero_shot_features, \
                   image_features, zero_shot_logits, logits
        else:
            return logits


@TRAINER_REGISTRY.register()
class MAILSRC_Trainer(TrainerX):
    def check_cfg(self, cfg):
        assert cfg.TRAINER.MAILSRC_Trainer.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)

        if cfg.TRAINER.MAILSRC_Trainer.PREC == "fp32" or cfg.TRAINER.MAILSRC_Trainer.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()


        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model)

        print("Turning off gradients in both the image and the text encoder")


        train_parameters = ['mail_learner']
        if cfg.TRAINER.MAILSRC_Trainer.PROJ:
            train_parameters.append('text_encoder.text_projection')
            train_parameters.append('image_encoder.proj')
        train_parameters.append('last_ln')
        train_parameters.append('last_adapter')


        flag = 0
        for name, param in self.model.named_parameters():
            for str in train_parameters:
                flag = 0
                if name.startswith(str) == True:
                    param.requires_grad_(True)
                    flag = 1
                    break
            if flag == 0:
                param.requires_grad_(False)
            if "ZS_image_encoder" in name:
                param.requires_grad_(False)

        # for name, param in self.model.named_parameters():
        #     if name_to_update not in name:
        #         # Make sure that VPT prompts are updated
        #         if "VPT" in name:
        #             param.requires_grad_(True)
        #         else:
        #             param.requires_grad_(False)

        # Double check
        enabled = set()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                enabled.add(name)
        # print(f"Parameters to be updated: {enabled}")

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        # NOTE: only give prompt_learner to the optimizer
        self.optim = build_optimizer(self.model, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("MultiModalMailSRCLearner", self.model, self.optim, self.sched)

        self.scaler = GradScaler() if cfg.TRAINER.MAILSRC_Trainer.PREC == "amp" else None

    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)

        model = self.model
        optim = self.optim
        scaler = self.scaler

        prec = self.cfg.TRAINER.MAILSRC_Trainer.PREC
        if prec == "amp":
            with autocast():
                loss_ce, normalized_text_features, zs_clip_text_embeddings, zs_image_embedd, image_ft, \
                zero_shot_logits, logits = model(image, label)
                # Calculate the L_SCL_text loss
                # loss_scl_text = F.l1_loss(normalized_text_features, zs_clip_text_embeddings.cuda(),
                #                           reduction='mean') * self.cfg.TRAINER.MAILSRC_Trainer.TEXT_LOSS_WEIGHT
                # # Calculate the L_SCL_image loss
                # loss_scl_image = F.l1_loss(image_ft, zs_image_embedd.cuda(),
                #                            reduction='mean') * self.cfg.TRAINER.MAILSRC_Trainer.IMAGE_LOSS_WEIGHT

                # loss_scl_image = (1 - torch.mean(F.cosine_similarity(image_ft, zs_image_embedd.cuda(),
                #                                                      dim=1))) * self.cfg.TRAINER.MAILSRC_Trainer.IMAGE_LOSS_WEIGHT
                # loss_scl_text = (1 - torch.mean(
                #     F.cosine_similarity(normalized_text_features, zs_clip_text_embeddings.cuda(),
                #                         dim=1))) * self.cfg.TRAINER.MAILSRC_Trainer.TEXT_LOSS_WEIGHT

                # Now calculate L_SCL_logits
                # L_SCL_logits = F.kl_div(
                #     F.log_softmax(logits / 1, dim=1),
                #     F.log_softmax(zero_shot_logits / 1, dim=1),
                #     reduction='sum',
                #     log_target=True
                # ) * (1 * 1) / logits.numel() * self.cfg.TRAINER.MAILSRC_Trainer.LOGIT_LOSS_WEIGHT
                # L_SCL = (L_SCL_logits + loss_scl_text + loss_scl_image)
                # loss = (loss_ce + L_SCL)
                loss = loss_ce
            optim.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
        else:
            loss_ce, normalized_text_features, zs_clip_text_embeddings, zs_image_embedd, image_ft, \
            zero_shot_logits, logits = model(image, label)
            # Calculate the L_SCL_text loss
            # loss_scl_text = F.l1_loss(normalized_text_features, zs_clip_text_embeddings.cuda(),
            #                           reduction='mean') * self.cfg.TRAINER.MAILSRC_Trainer.TEXT_LOSS_WEIGHT
            # # Calculate the L_SCL_image loss
            # loss_scl_image = F.l1_loss(image_ft, zs_image_embedd.cuda(),
            #                            reduction='mean') * self.cfg.TRAINER.MAILSRC_Trainer.IMAGE_LOSS_WEIGHT

            # loss_scl_image = (1 - torch.mean(F.cosine_similarity(image_ft, zs_image_embedd.cuda(), dim=1))) * self.cfg.TRAINER.MAILSRC_Trainer.IMAGE_LOSS_WEIGHT
            # loss_scl_text = (1 - torch.mean(F.cosine_similarity(normalized_text_features, zs_clip_text_embeddings.cuda(),
            #                                                    dim=1))) * self.cfg.TRAINER.MAILSRC_Trainer.TEXT_LOSS_WEIGHT
            #
            # # Now calculate L_SCL_logits
            # L_SCL_logits = F.kl_div(
            #     F.log_softmax(logits / 1, dim=1),
            #     F.log_softmax(zero_shot_logits / 1, dim=1),
            #     reduction='sum',
            #     log_target=True
            # ) * (1 * 1) / logits.numel() * self.cfg.TRAINER.MAILSRC_Trainer.LOGIT_LOSS_WEIGHT
            # L_SCL = (L_SCL_logits + loss_scl_text + loss_scl_image)
            # loss = (loss_ce + L_SCL)
            loss = loss_ce
            optim.zero_grad()
            loss.backward()
            optim.step()

        loss_summary = {"loss": loss.item(), 'acc': compute_accuracy(logits, label)[0].item()}

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def get_gauss(self, mu, sigma):
        gauss = lambda x: (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
        return gauss

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]

        input = input.to(self.device)
        label = label.to(self.device)
        return input, label

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            # Ignore fixed token vectors
            if "prompt_learner.token_prefix" in state_dict:
                del state_dict["prompt_learner.token_prefix"]

            if "mail_learner.embedding" in state_dict:
                del state_dict["mail_learner.embedding"]

            if "prompt_learner.token_suffix" in state_dict:
                del state_dict["prompt_learner.token_suffix"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)
