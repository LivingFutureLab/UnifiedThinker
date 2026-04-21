import os
import math
import torch
import torch.nn as nn
import numpy as np
from PIL import Image


def calculate_shift(
    image_seq_len,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.16,
):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu


class BaseTrainer:
    """
    基础训练器类，用于训练模型。

    属性:
        accelerator: 加速器对象，用于分布式训练。
        model_dict: 包含模型的字典。
        optimizer: 优化器对象。
        lr_scheduler: 学习率调度器。
        loss: 损失函数。
        device: 设备（CPU或GPU）。
        weight_dtype: 权重数据类型。
        args: 训练参数。
    """

    def __init__(
        self,
        args,
        accelerator,
        model_dict,
        optimizer,
        lr_scheduler,
        weight_dtype,
        device,
        noise_scheduler="lognorm",
    ):
        """
        初始化 BaseTrainer 类。

        参数:
            args: 训练参数。
            accelerator: 加速器对象。
            model_dict: 包含模型的字典。
            optimizer: 优化器对象。
            lr_scheduler: 学习率调度器。
            weight_dtype: 权重数据类型。
            device: 设备（CPU或GPU）。
            loss_fn: 损失函数名称。
        """
        self.accelerator = accelerator
        self.model_dict = model_dict
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.noise_scheduler = noise_scheduler
        self.device = device

        self.weight_dtype = weight_dtype
        self.args = args
        self._set_params_to_optimize()

    def _set_params_to_optimize(self):
        """
        设置需要优化的参数列表。
        """
        self.params_to_optimize = [
            p
            for p in filter(
                lambda p: p.requires_grad, self.model_dict["train_model"].parameters()
            )
        ]

    @torch.no_grad()
    def _vae_encode(self, x):
        """
        对输入进行 VAE 编码。

        参数:
            x: 输入normalized tensor。

        返回:
            latents: 编码后的潜在变量。
        """
        x = x.to(dtype=self.model_dict["vae"].dtype)
        latents = []
        for i in range(0, x.shape[0], self.args.vae_encode_batch_size):
            latents.append(
                self.model_dict["vae"]
                .encode(x[i : i + self.args.vae_encode_batch_size])
                .latent_dist.sample()
            )
        latents = torch.cat(latents, dim=0)
        latents = (
            latents - self.model_dict["vae"].config.shift_factor
        ) * self.model_dict["vae"].config.scaling_factor
        latents = latents.to(dtype=self.weight_dtype)
        return latents

    @torch.no_grad()
    def _vae_decode(self, latents, positive=False):
        """
        对潜在变量进行 VAE 解码。

        参数:
            latents: 潜在变量张量。
            positive: 是否将输出限制为正值。

        返回:
            x: 解码后的图像张量。
        """
        latents = (
            latents.to(dtype=self.model_dict["vae"].dtype)
            / self.model_dict["vae"].config.scaling_factor
            + self.model_dict["vae"].config.shift_factor
        )
        x = []
        for i in range(0, latents.shape[0], self.args.vae_encode_batch_size):
            x.append(
                self.model_dict["vae"]
                .decode(latents[i : i + self.args.vae_encode_batch_size])
                .sample
            )
        x = torch.cat(x, dim=0)
        x = x.to(dtype=self.weight_dtype)
        x = torch.clamp(x, -1, 1)
        if positive:
            x = (x + 1) / 2
        return x

    @torch.no_grad()
    def _conditioner_encode(self, captions):
        """
        对输入的文本提示进行编码。

        参数:
            captions: 文本提示列表。

        返回:
            input_condition: 包含编码结果的字典。
        """
        bs = len(captions)

        pooled_prompt_embeds = BaseTrainer._encode_prompt_with_clip(
            text_encoder=self.model_dict["text_encoder"][0],
            tokenizer=self.model_dict["tokenizer"][0],
            prompt=captions,
            device=self.device,
        )
        # print(f"[Clip Embed] clip embed shape : {pooled_prompt_embeds.shape}")

        prompt_embeds = BaseTrainer._encode_prompt_with_t5(
            text_encoder=self.model_dict["text_encoder"][1],
            tokenizer=self.model_dict["tokenizer"][1],
            max_sequence_length=512,
            prompt=captions,
            device=self.device,
        )
        # print(f"[T5 Embed] T5 embed shape : {prompt_embeds.shape}")

        text_ids = torch.zeros(bs, prompt_embeds.shape[1], 3).to(
            device=self.device, dtype=self.weight_dtype
        )

        input_condition = {
            "pooled_projections": pooled_prompt_embeds,
            "encoder_hidden_states": prompt_embeds,
            "txt_ids": text_ids,
        }

        return input_condition

    def get_sigma(self, bsz):
        """
        获取 sigma 和时间步长。

        参数:
            bsz: 批次大小。

        返回:
            tuple: 包含 sigma 和时间步长的元组。
        """
        if self.noise_scheduler == "uniform":
            sigmas = torch.rand((bsz,), device=self.device)
            timesteps = sigmas * 1000.0
            sigmas = sigmas.view(-1, 1, 1, 1).to(dtype=self.weight_dtype)
        elif "lognorm" in self.noise_scheduler:
            sigmas = torch.sigmoid(torch.randn((bsz,), device=self.device))
            timesteps = sigmas * 1000.0
            sigmas = sigmas.view(-1, 1, 1, 1).to(dtype=self.weight_dtype)
        return sigmas, timesteps

    def apply_flow_schedule_shift(self, sigmas, image_seq_len):
        # Resolution-dependent shift value calculation used by official Flux inference implementation
        mu = calculate_shift(image_seq_len, 256, 4096, 0.5, 1.16)
        shift = math.exp(mu)
        sigmas = (sigmas * shift) / (1 + (shift - 1) * sigmas)
        return sigmas

    def _train_step(self, batch):
        """
        执行一次训练步骤。

        参数:
            batch: 输入批次数据。

        返回:
            return_loss: 包含损失的字典。
        """
        raise NotImplementedError("_train_step 方法需要在子类中实现")

    def train_step(self, batch, gstep=None):
        """
        执行训练步骤并更新模型参数。

        参数:
            batch: 输入批次数据。
            gstep: 当前训练步数（可选）。

        返回:
            loss: 包含损失的字典。
        """
        self.gstep = gstep

        loss = self._train_step(batch)

        self.accelerator.backward(loss["loss"])
        if self.accelerator.sync_gradients:
            total_norm = self.accelerator.clip_grad_norm_(
                self.params_to_optimize, self.args.max_grad_norm
            )
            # loss["grad_norm"] = total_norm
            # if loss["grad_norm"] is not None:
            #     loss["grad_norm"] = loss["grad_norm"].detach().item()
            if total_norm is not None:
                if torch.is_tensor(total_norm):
                    loss["grad_norm"] = total_norm.detach().item()
                else:
                    loss["grad_norm"] = total_norm
            
        self.optimizer.step()
        self.lr_scheduler.step()
        self.optimizer.zero_grad(set_to_none=True)

        loss["loss"] = loss["loss"].detach().item()
        return loss

    def log_train_info(self, batch, save_dir, global_update_step):
        image_dir = os.path.join(save_dir, "train_images")
        os.makedirs(image_dir, exist_ok=True)
        ## save image and mask side by side
        pixel_values = batch["pixel_values"]  # BCHW
        pixel_values = pixel_values.to(dtype=torch.float32).cpu().numpy()
        pixel_values = (pixel_values + 1) * 127.5
        pixel_values = pixel_values.astype(np.uint8)
        pixel_values = np.transpose(pixel_values, (0, 2, 3, 1))  # BHWC

        mask = batch["mask"]  # B1HW
        mask = mask.to(dtype=torch.float32).cpu().numpy()
        mask = mask * 255
        mask = mask.astype(np.uint8)
        mask = np.transpose(mask, (0, 2, 3, 1))  # BHW1
        mask = np.repeat(mask, 3, axis=3)  # BHW3

        # 先在宽度维度上拼接每个batch的图像和mask
        combined_batch = np.concatenate([pixel_values, mask], axis=2)  # B(H)(2W)3
        # 在高度维度上拼接所有batch
        combined_image = np.concatenate(list(combined_batch), axis=0)  # (BH)(2W)3
        combined_pil = Image.fromarray(combined_image)
        combined_pil.save(os.path.join(image_dir, f"{global_update_step}.png"))

        ## save caption
        caption_path = os.path.join(image_dir, f"{global_update_step}.txt")
        with open(caption_path, "w", encoding="utf-8") as f:
            f.write("\n".join(batch["caption"]))

    def save_model(self, save_path):
        """
        保存模型到指定路径, 目前为空实现。

        参数:
            save_path: 保存路径。
        """
        raise NotImplementedError("save_model 方法需要在子类中实现")

    def load_model(self, load_path):
        """
        从指定路径加载模型, 目前为空实现。

        参数:
            load_path: 加载路径。
        """
        raise NotImplementedError("load_model 方法需要在子类中实现")

    @staticmethod
    def _encode_prompt_with_t5(
        text_encoder,
        tokenizer,
        max_sequence_length=512,
        prompt=None,
        num_images_per_prompt=1,
        device=None,
        text_input_ids=None,
    ):
        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        if tokenizer is not None:
            text_inputs = tokenizer(
                prompt,
                padding="max_length",
                max_length=max_sequence_length,
                truncation=True,
                return_length=False,
                return_overflowing_tokens=False,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids
        else:
            if text_input_ids is None:
                raise ValueError(
                    "text_input_ids must be provided when the tokenizer is not specified"
                )

        prompt_embeds = text_encoder(text_input_ids.to(device))[0]

        dtype = text_encoder.dtype
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

        _, seq_len, _ = prompt_embeds.shape

        # duplicate text embeddings and attention mask for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(
            batch_size * num_images_per_prompt, seq_len, -1
        )

        return prompt_embeds

    @staticmethod
    def _encode_prompt_with_clip(
        text_encoder,
        tokenizer,
        prompt: str,
        device=None,
        text_input_ids=None,
        num_images_per_prompt: int = 1,
    ):
        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        if tokenizer is not None:
            text_inputs = tokenizer(
                prompt,
                padding="max_length",
                max_length=77,
                truncation=True,
                return_overflowing_tokens=False,
                return_length=False,
                return_tensors="pt",
            )

            text_input_ids = text_inputs.input_ids
        else:
            if text_input_ids is None:
                raise ValueError(
                    "text_input_ids must be provided when the tokenizer is not specified"
                )

        prompt_embeds = text_encoder(
            text_input_ids.to(device), output_hidden_states=False
        )

        # Use pooled output of CLIPTextModel
        prompt_embeds = prompt_embeds.pooler_output
        prompt_embeds = prompt_embeds.to(dtype=text_encoder.dtype, device=device)

        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, -1)

        return prompt_embeds
