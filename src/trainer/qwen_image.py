import os
import torch
from src.trainer.base_trainer import BaseTrainer
import torch.distributed as dist
from src.pipe.pipeline_qwen_image import calculate_shift
import time
import shutil
from safetensors.torch import load_model, save_model, save_file
from datetime import datetime
from src.utils.env_utils import in_notebook

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

    def _vae_encode(self, pixel_values):
        pixel_values = pixel_values.unsqueeze(2)
        latents = []
        for i in range(0, pixel_values.shape[0], self.args.vae_encode_batch_size):
            latents.append(
                self.model_dict["vae"]
                .encode(pixel_values[i : i + self.args.vae_encode_batch_size].to(self.device, dtype=self.model_dict["vae"].dtype))
                .latent_dist.mode()
            )
        latents = torch.cat(latents, dim=0)
        latents_mean = (
            torch.tensor(self.model_dict["vae"].config.latents_mean)
            .view(1, self.model_dict["vae"].config.z_dim, 1, 1, 1)
            .to(latents.device, latents.dtype)
        )
        latents_std = 1.0 / torch.tensor(self.model_dict["vae"].config.latents_std).view(1, self.model_dict["vae"].config.z_dim, 1, 1, 1).to(
            latents.device, latents.dtype
        )
        latents = (latents - latents_mean) * latents_std
        return latents
    
    def _vae_decode(self, latents):
        latents = latents.to(self.model_dict["vae"].dtype)
        latents_mean = (
            torch.tensor(self.model_dict["vae"].config.latents_mean)
            .view(1, self.model_dict["vae"].config.z_dim, 1, 1, 1)
            .to(latents.device, latents.dtype)
        )
        latents_std = 1.0 / torch.tensor(self.model_dict["vae"].config.latents_std).view(1, self.model_dict["vae"].config.z_dim, 1, 1, 1).to(
            latents.device, latents.dtype
        )
        latents = latents / latents_std + latents_mean
        image = self.model_dict["vae"].decode(latents, return_dict=False)[0][:, :, 0]
        image = self.model_dict["pipe"].image_processor.postprocess(image, output_type="pt")
        return image
    
    def _train_step_gen(self, batch):
        ##############################
        # 提取图片特征
        pixel_values = batch["pixel_values"].to(self.device, dtype=self.weight_dtype)
        prompt_embeds, prompt_embeds_mask = self.model_dict["pipe"].encode_prompt(
                prompt=batch["prompt"],
                device=self.device,
                max_sequence_length=512 # Maximum sequence length to use with the `prompt`.
            )
        # print('prompt_embeds', prompt_embeds.shape) 
        # prompt_embeds torch.Size([1, 11, 3584])
        
        gt_latents = self._vae_encode(pixel_values)
        _, _, image_h, image_w = batch["pixel_values"].shape
        bsz, num_channels_latents, _, latent_h, latent_w = gt_latents.shape
        # bsz, channels_latents, num_frames, latent_h, latent_w = gt_latents.shape
        # print("gt_latents", gt_latents.shape) # gt_latents torch.Size([1, 16, 1, 166, 166])
        gt_latents = self.model_dict["pipe"]._pack_latents(gt_latents, bsz, num_channels_latents, latent_h, latent_w)
        # print("gt_latents", gt_latents.shape) # gt_latents torch.Size([1, 6889, 64])
        
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
        noise = torch.randn_like(gt_latents)
        noisy_latent = (1 - sigmas) * gt_latents + sigmas * noise
        # print(noisy_latent.shape) # torch.Size([1, 6889, 64])
        img_shapes = [(1, latent_h // 2, latent_w // 2)] * bsz
        # print(len(img_shapes), img_shapes) # 1 [(1, 83, 83)]
        
        model_pred = self.model_dict["transformer"](
                hidden_states=noisy_latent.to(self.weight_dtype),
                timestep=sigmas,
                encoder_hidden_states_mask=prompt_embeds_mask,
                encoder_hidden_states=prompt_embeds,
                img_shapes=img_shapes,
                txt_seq_lens=prompt_embeds_mask.sum(dim=1).tolist(),
                return_dict=False,
            )[0]

        target = noise - gt_latents

        # Compute regular loss.
        loss = torch.mean(
            ((model_pred.float() - target.float()) ** 2).reshape(target.shape[0], -1),
            1,
        )
        loss = loss.mean()
        return loss
    
    def _train_step_und(self, batch):
        for k in batch:
            batch[k] = batch[k].to(device=self.device)
            if k not in ["input_ids", "labels", "image_grid_thw"]:
                batch[k] = batch[k].to(dtype=self.weight_dtype)    
        outputs = self.model_dict['text_encoder'](**batch)
        loss = outputs.loss
        return loss

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
        
        if "gen" in batch and "und" in batch:
            batch_gen = batch["gen"]
            batch_und = batch["und"]
            loss_gen = self._train_step_gen(batch_gen)
            loss_und = self._train_step_und(batch_und)
            loss = loss_gen + loss_und
            loss = {"loss": loss, 
                    "loss_gen": loss_gen.detach().item(), 
                    "loss_und": loss_und.detach().item()}
        elif "gen" in batch:
            batch_gen = batch["gen"]
            loss_gen = self._train_step_gen(batch_gen)
            loss = loss_gen
            loss = {"loss": loss, 
                    "loss_gen": loss_gen.detach().item()}
        elif "und" in batch:
            batch_und = batch["und"]
            loss_und = self._train_step_und(batch_und)
            loss = loss_und
            loss = {"loss": loss, 
                    "loss_und": loss_und.detach().item()}
        else:
            raise ValueError()

        self.accelerator.backward(loss["loss"])
        if self.accelerator.sync_gradients:
            total_norm = self.accelerator.clip_grad_norm_(
                self.params_to_optimize, self.args.max_grad_norm
            )
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
    
    def get_caption_language(self, prompt):
        ranges = [
            ('\u4e00', '\u9fff'),  # CJK Unified Ideographs
            # ('\u3400', '\u4dbf'),  # CJK Unified Ideographs Extension A
            # ('\u20000', '\u2a6df'), # CJK Unified Ideographs Extension B
        ]
        for char in prompt:
            if any(start <= char <= end for start, end in ranges):
                return 'zh'
        return 'en'

    @torch.no_grad()
    def generate_images(
        self,
        pipe,
        prompts,
        output_dir,
        start_idx=0,
    ):
        os.makedirs(output_dir, exist_ok=True)

        positive_magic = {
            "en": "Ultra HD, 4K, cinematic composition.",
            "zh": "超清，4K，电影级构图",
        }

        for i, prompt in enumerate(prompts):
            # prompt = prompt.replace("「", '"').replace("」", '"').replace("[sep]", " ")
            print("Start processing prompt: ", prompt)
            image = pipe(
                prompt=prompt + positive_magic[self.get_caption_language(prompt)],
                negative_prompt=" ",
                width=1328,
                height=1328,
                num_inference_steps=50,
                true_cfg_scale=4.0,
                generator=torch.Generator(device="cuda").manual_seed(42),
            ).images[0]

            for j, img in enumerate([image]):
                is_exist = os.path.exists(
                    os.path.join(output_dir, f"{start_idx + i + 1}_{j + 1}.png")
                )
                print(
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),  # 打印当前时间
                    os.path.join(output_dir, f"{start_idx + i + 1}_{j + 1}.png"),
                    is_exist,
                )
                if not is_exist:
                    img.save(os.path.join(output_dir, f"{start_idx + i + 1}_{j + 1}.png"))

        
    @torch.no_grad()
    def validate(self, save_dir, args):
        
        PROMPTS_LIST = []
        with open('/data/oss_bucket_0/litong/data/2-train-valid/for_train_valid.txt', "r", encoding="utf-8") as f:
            PROMPTS_LIST.extend([
                line.strip() for line in f.readlines() if line.strip()
            ])
        
        # 分割 prompt
        num_devices = args.gpu.total
        rank = args.gpu.index
        chunk_size = len(PROMPTS_LIST) // num_devices

        start_idx = rank * chunk_size
        end_idx = (rank + 1) * chunk_size if rank < num_devices - 1 else None
        prompt_list = PROMPTS_LIST[start_idx:end_idx]
        # prompt_list = prompt_list[:0]
        print("prompt_list", len(prompt_list))

        print("start sampling")
        # img_save_root = os.path.join(save_dir, "images")
        self.generate_images( 
                    pipe = self.model_dict["pipe"],
                    prompts = prompt_list,
                    output_dir = save_dir,
                    start_idx = start_idx,
                    )

    def save_model(self, save_path):
        if in_notebook():
            tmp_save_path = save_path
        else:
            assert save_path.startswith("oss://tstar-image-dataset/"), "wrong of model_path: {}".format(save_path)
            tmp_save_path = os.path.join("./tmp_ckpt", save_path.replace("oss://tstar-image-dataset/", ""))
        
        os.makedirs(tmp_save_path, exist_ok=True)
        train_model = self.accelerator.unwrap_model(self.model_dict["train_model"])
        
        # 对于 Stage 2, `get_state_dict` 也是安全的，它会直接返回 model.state_dict()
        # 对于 Stage 3, `get_state_dict` 是必须的，它会从所有 rank 收集参数
        full_state_dict = self.accelerator.get_state_dict(self.model_dict["train_model"])
        
        trainable_param_names = {name for name, param in train_model.named_parameters() if param.requires_grad}
        state_to_save = {name: param for name, param in full_state_dict.items() if name in trainable_param_names}
        
        if self.accelerator.is_main_process:
            output_model_file = os.path.join(tmp_save_path, "pytorch_model.bin")
            self.accelerator.save(state_to_save, output_model_file)
        print(f"Save one checkpoint to {tmp_save_path} successfully!")
        
        if not in_notebook():
            from src.model.utils import upload_model_weight_oss
            upload_model_weight_oss(tmp_save_path, save_path)
            print(f"Upload checkpoint to {save_path} successfully!")
            try:
                shutil.rmtree(tmp_save_path)
                print(f"Delete the temporary directory: {tmp_save_path}")
            except OSError as e:
                print(f"Wrong when delete the temporary directory: {tmp_save_path}. {e}")        

    def load_model(self, load_path):
        pass

    def log_train_info(self, batch, save_dir, global_update_step):
        # 输入特征，无需记录
        pass
