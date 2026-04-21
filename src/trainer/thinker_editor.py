#coding=utf-8
import os
import torch
from src.trainer.base_trainer import BaseTrainer
import torch.distributed as dist
from src.pipe.pipeline_qwen_image import calculate_shift
import time
import shutil
from PIL import Image
#from safetensors.torch import load_model, save_model, save_file
from datetime import datetime
from src.utils.env_utils import in_notebook
from src.pipe.pipeline_qwen_image_edit_plus import CONDITION_IMAGE_SIZE, VAE_IMAGE_SIZE, calculate_dimensions

def get_caption_language(prompt):
    ranges = [
        ('\u4e00', '\u9fff'),  # CJK Unified Ideographs
        # ('\u3400', '\u4dbf'),  # CJK Unified Ideographs Extension A
        # ('\u20000', '\u2a6df'), # CJK Unified Ideographs Extension B
    ]
    for char in prompt:
        if any(start <= char <= end for start, end in ranges):
            return 'zh'
    return 'en'

class Trainer(BaseTrainer):
    def __init__(self, *wargs, **kwargs):
        super().__init__(*wargs, **kwargs)
        
    def gather(self, tensor):
        tensor = tensor.detach().clone()
        tensor_all = torch.empty(
            self.args.dist.world_size * tensor.numel(),
            dtype=tensor.dtype,
            device=tensor.device
        )
        dist.all_gather_into_tensor(tensor_all, tensor)
        return tensor_all.view(-1, *tensor.size()[1:])

    def _train_step_gen_t2i(self, batch):
        images = batch["image"]
        assert len(images) == 1, "right now support batch_size 1"
        image_width_tgt, image_height_tgt = images[0].size
        if False:
            # resize 放到 dataset 里面
            image_width_tgt, image_height_tgt = calculate_dimensions(VAE_IMAGE_SIZE, image_width_tgt / image_height_tgt)
        image_tgt = self.model_dict["pipe"].image_processor.preprocess(images[0], image_height_tgt, image_width_tgt).unsqueeze(2)   # [1, 3, 1, 1024, 1024]
        latents_tgt = self.model_dict["pipe"]._encode_vae_image(image_tgt.to(self.device, dtype=self.weight_dtype), None)           # torch.Size([1, 16, 1, 128, 128])
        bsz, num_channels_latents, _, latent_h, latent_w = latents_tgt.shape
        latents_tgt = self.model_dict["pipe"]._pack_latents(latents_tgt, bsz, num_channels_latents, latent_h, latent_w)             # torch.Size([1, 4096, 64])
        
        prompt_embeds, prompt_embeds_mask = self.model_dict["pipe"].encode_prompt(
            prompt=batch["prompt"],
            device=self.device,
            max_sequence_length=512, # Maximum sequence length to use with the `prompt`
            prompt_cot=batch["prompt_cot"]  # for thinking
        )
        
        # Prepare schedule
        ## sigmas(0-1), timesteps(0-1000)
        sigmas, timesteps = self.get_sigma(bsz)
        if "shift" in self.noise_scheduler:
            image_seq_len = (latent_h // 2) * (latent_w // 2)
            mu = calculate_shift(
                image_seq_len,
                self.model_dict["pipe"].scheduler.config.get("base_image_seq_len"),
                self.model_dict["pipe"].scheduler.config.get("max_image_seq_len"),
                self.model_dict["pipe"].scheduler.config.get("base_shift"),
                self.model_dict["pipe"].scheduler.config.get("max_shift"),
            )
            sigmas = self.model_dict["pipe"].scheduler.time_shift(mu, 1.0, sigmas)
            timesteps = (sigmas * 1000.0).view(-1)

        # print(sigmas.shape) # torch.Size([1, 1, 1, 1])
        sigmas = sigmas.view(-1)

        ## Noised latent
        noise = torch.randn_like(latents_tgt)
        noisy_latent = (1 - sigmas) * latents_tgt + sigmas * noise
        img_shapes = [(1, latent_h // 2, latent_w // 2)] * bsz
        
        model_pred = self.model_dict["dit"](
                hidden_states=noisy_latent.to(self.weight_dtype),
                timestep=sigmas,
                encoder_hidden_states_mask=prompt_embeds_mask,
                encoder_hidden_states=prompt_embeds,
                img_shapes=img_shapes,
                txt_seq_lens=prompt_embeds_mask.sum(dim=1).tolist(),
                return_dict=False,
            )[0]

        target = noise - latents_tgt

        # Compute regular loss.
        loss = torch.mean(
            ((model_pred.float() - target.float()) ** 2).reshape(target.shape[0], -1),
            1,
        )
        loss = loss.mean()
        return loss
    
    def _train_step_gen_edit(self, batch):
        # prompt
        raw_condition_images = batch["raw_condition_images"]    # [ list of PIL.Image, ]
        assert len(raw_condition_images) == 1, "right now support batch_size 1"
        batch_size = 1
        raw_condition_images = raw_condition_images[0]
        
        condition_image_sizes = []
        condition_images = []
        vae_image_sizes = []
        vae_images = []
        for img in raw_condition_images:
            image_width, image_height = img.size
            condition_width, condition_height = calculate_dimensions(
                CONDITION_IMAGE_SIZE, image_width / image_height
            )
            if False:   # resize 放到 dataset 里面
                vae_width, vae_height = calculate_dimensions(VAE_IMAGE_SIZE, image_width / image_height)
            else:
                vae_width, vae_height = image_width, image_height
                
            condition_image_sizes.append((condition_width, condition_height))
            vae_image_sizes.append((vae_width, vae_height))
            condition_images.append(self.model_dict["pipe"].image_processor.resize(img, condition_height, condition_width))
            vae_images.append(self.model_dict["pipe"].image_processor.preprocess(img, vae_height, vae_width).unsqueeze(2))
        
        prompt_embeds, prompt_embeds_mask = self.model_dict["pipe"].encode_prompt(
            image=condition_images,
            prompt=batch["prompt"],
            device=self.device,
            max_sequence_length=1024,
            prompt_cot=batch["prompt_cot"]  # for thinking
        )                                                               # torch.Size([1, 212, 3584])
        
        # groundtruth
        images = batch["image"]
        assert len(images) == 1, "right now support batch_size 1"
        image_width_tgt, image_height_tgt = images[0].size
        if False: # resize 放到 dataset 里面
            image_width_tgt, image_height_tgt = calculate_dimensions(VAE_IMAGE_SIZE, image_width_tgt / image_height_tgt)
        image_tgt = self.model_dict["pipe"].image_processor.preprocess(images[0], image_height_tgt, image_width_tgt).unsqueeze(2)
        latents_tgt = self.model_dict["pipe"]._encode_vae_image(image_tgt.to(self.device, dtype=self.weight_dtype), None)
        bsz, num_channels_latents, _, latent_h, latent_w = latents_tgt.shape
        latents_tgt = self.model_dict["pipe"]._pack_latents(latents_tgt, bsz, num_channels_latents, latent_h, latent_w)     # torch.Size([1, 4096, 64])
            
        num_channels_latents = self.model_dict["pipe"].transformer.config.in_channels // 4
        latents_tgt, image_latents = self.model_dict["pipe"].prepare_latents(
            vae_images,
            batch_size,
            num_channels_latents,
            image_height_tgt,
            image_width_tgt,
            prompt_embeds.dtype,
            self.device,
            None,
            latents_tgt
        )
        
        # Prepare schedule
        ## sigmas(0-1), timesteps(0-1000)
        sigmas, timesteps = self.get_sigma(batch_size)
        if "shift" in self.noise_scheduler:
            image_seq_len = (latent_h // 2) * (latent_w // 2)
            mu = calculate_shift(
                image_seq_len,
                self.model_dict["pipe"].scheduler.config.get("base_image_seq_len"),
                self.model_dict["pipe"].scheduler.config.get("max_image_seq_len"),
                self.model_dict["pipe"].scheduler.config.get("base_shift"),
                self.model_dict["pipe"].scheduler.config.get("max_shift"),
            )
            sigmas = self.model_dict["pipe"].scheduler.time_shift(mu, 1.0, sigmas)
            timesteps = (sigmas * 1000.0).view(-1)

        # print(sigmas.shape) # torch.Size([1, 1, 1, 1])
        sigmas = sigmas.view(-1)

        ## Noised latent
        noise = torch.randn_like(latents_tgt)
        noisy_latent = (1 - sigmas) * latents_tgt + sigmas * noise
        
        img_shapes = [
            [
                (1, latent_h // 2, latent_w // 2),
                *[
                    (1, vae_height // self.model_dict["pipe"].vae_scale_factor // 2, vae_width // self.model_dict["pipe"].vae_scale_factor // 2)
                    for vae_width, vae_height in vae_image_sizes
                ],
            ]
        ] * batch_size                  # [[(1, 64, 64), (1, 78, 52)]]
        
        latent_model_input = noisy_latent
        if image_latents is not None:
            latent_model_input = torch.cat([noisy_latent, image_latents], dim=1)
        
        # import pdb; pdb.set_trace()
        model_pred = self.model_dict["dit"](
                hidden_states=latent_model_input.to(self.weight_dtype),
                timestep=sigmas,
                encoder_hidden_states_mask=prompt_embeds_mask,
                encoder_hidden_states=prompt_embeds,
                img_shapes=img_shapes,
                txt_seq_lens=prompt_embeds_mask.sum(dim=1).tolist(),
                return_dict=False,
            )[0]
        model_pred = model_pred[:, : noisy_latent.size(1)]

        target = noise - latents_tgt

        # Compute regular loss.
        loss = torch.mean(
            ((model_pred.float() - target.float()) ** 2).reshape(target.shape[0], -1),
            1,
        )
        loss = loss.mean()
        return loss
        
    def _train_step_gen(self, batch):
        if len(batch["raw_condition_images"][0]) == 0:
            return self._train_step_gen_t2i(batch), "t2i"
        else:
            return self._train_step_gen_edit(batch), "edit"
                
    def _train_step_und(self, batch):
        record_ids = batch.pop("record_ids", None)
        import pdb; pdb.set_trace()
        for k in batch:
            batch[k] = batch[k].to(device=self.device)
            if k not in ["input_ids", "labels", "image_grid_thw"]:
                batch[k] = batch[k].to(dtype=self.weight_dtype)    
        outputs = self.model_dict['thinker'](**batch)
        loss = outputs.loss
        
        #if torch.isnan(loss): # TODO: check
        if not torch.isfinite(loss):
            print(f"\n[Warning] Understanding loss is NaN or inf, resetting to 0.0.")
            print(f"  - Record IDs: {record_ids}")
            #loss = torch.tensor(0.0, device=self.device, requires_grad=True)
            
            # 确保 loss 张量与模型的计算图保持连接
            if hasattr(outputs, 'logits') and outputs.logits is not None and (not torch.any(torch.isfinite(outputs.logits))):
                loss = torch.mean(outputs.logits) * 0.0
            else:
                params = self.model_dict['thinker'].parameters()
                loss = sum(p.sum() for p in params) * 0.0
            
        return loss
          
    
    def train_step(self, batch, gstep=None):
        """
        当两个loss都存在时, 共享的text_encoder在一次backward中会被访问两次, 配合gradient_checkpointing会导致梯度被reduce两次。
        
        我们使用 accelerator.no_sync 来手动控制梯度同步。
        
        解决方案:
            1) 仅在必要时（即 gen 和 und 两个分支都存在时） 使用 no_sync。
            2) 在 no_sync 上下文中，对其中一个损失 (例如 loss_und) 进行反向传播。这只会在每个 GPU 本地计算并累积梯度，不会进行跨卡同步。
            3) 在上下文之外, 对另一个损失 (loss_gen) 进行反向传播。这次调用会计算它自己的梯度，将其与之前累积的梯度相加，然后触发一次性的、针对所有累积梯度的跨卡同步。
            为了让计算图在第一次 backward 后不被释放，我们需要在 no_sync 块内的 backward 调用中设置 retain_graph=True。
        """
        self.gstep = gstep
        
        total_loss_for_logging = 0.0
        loss_dict_for_sync = {}
        
        assert "gen" in batch and "und" in batch, "Batch must contain 'gen' and 'und' data."

        # inputs -> editor -> loss
        loss_gen, loss_key = self._train_step_gen(batch["gen"])
        total_loss_for_logging += loss_gen
        loss_dict_for_sync[f"loss_gen_{loss_key}"] = loss_gen
        
        # inputs -> thinker -> loss
        import pdb; pdb.set_trace()
        loss_und = self._train_step_und(batch["und"])
        weighted_loss_und = loss_und * self.args.loss_weight_und
        total_loss_for_logging += weighted_loss_und
        loss_dict_for_sync["loss_und"] = loss_und
        
        loss_dict_for_sync["loss"] = total_loss_for_logging
        
        # import pdb; pdb.set_trace()
        with self.accelerator.no_sync(self.model_dict["train_model"]):
            # 第一次反向传播，只在本地累积梯度，不进行同步。retain_graph=True 是必需的，因为它保留了计算图，使得第二次反向传播（针对loss_gen）可以继续使用。
            self.accelerator.backward(weighted_loss_und, retain_graph=True)

        # 第二次（也是最后一次）反向传播。这次调用在 no_sync 上下文之外，它会计算 loss_gen 的梯度，
        # 将其与之前累积的梯度（来自loss_und）相加，然后触发一次针对所有参数总梯度的同步（reduction）。
        self.accelerator.backward(loss_gen)
  
        grad_norm_tensor = None
        if self.accelerator.sync_gradients:
            # 梯度裁剪发生在此处，此时所有梯度已经正确地计算和同步完毕。
            grad_norm_tensor = self.accelerator.clip_grad_norm_(
                self.params_to_optimize, self.args.max_grad_norm
            )
        
        self.optimizer.step()
        self.lr_scheduler.step()
        self.optimizer.zero_grad(set_to_none=True)

        synced_losses = self.accelerator.gather_for_metrics(loss_dict_for_sync)
        loss_metrics = {k: v.mean().detach().item() for k, v in synced_losses.items()}
        if grad_norm_tensor is not None:
            if torch.is_tensor(grad_norm_tensor):
                loss_metrics["grad_norm"] = grad_norm_tensor.detach().item()
            else:
                loss_metrics["grad_norm"] = grad_norm_tensor      
        loss_metrics["loss_weight_und"] = self.args.loss_weight_und
        return loss_metrics
    