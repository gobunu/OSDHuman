import os
import time

import yaml
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, CLIPTextModel
from diffusers import DDPMScheduler
from models.autoencoder_kl import AutoencoderKL
from models.unet_2d_condition import UNet2DConditionModel
from peft import LoraConfig
import pickle
import numpy as np
from PIL import Image
import torchvision.transforms as T

from my_utils.vaehook import VAEHook, perfcount


def save_tensor_as_images(tensor, filename_prefix):
    # Assume the tensor shape is [B, C, H, W].
    B, C, H, W = tensor.size()

    # Ensure the output directory exists.
    if not os.path.exists('./'):
        os.makedirs('./')

    # Use bicubic resizing to upsample each channel preview.
    sinc_resize = T.Resize((512, 512), interpolation=T.InterpolationMode.BICUBIC)

    for c in range(C):
        # Extract a single channel.
        channel_tensor = tensor[:, c, :, :]  # [B, H, W]

        # Convert the tensor to a NumPy array before creating a PIL image.
        channel_image = channel_tensor.squeeze().cpu().numpy()

        # Normalize floating-point data into the [0, 255] range before casting.
        if channel_image.dtype != np.uint8:
            channel_image = (channel_image - channel_image.min()) / (channel_image.max() - channel_image.min()) * 255
            channel_image = channel_image.astype(np.uint8)

        # Save the preview as a grayscale image.
        channel_image = Image.fromarray(channel_image).convert('L')

        # Resize the channel preview to 512x512.
        channel_image_resized = sinc_resize(channel_image)

        # Write the channel preview to disk.
        save_path = f"{filename_prefix}_{c}.png"
        channel_image_resized.save(save_path)
        print(f"Saved: {save_path}")


def initialize_vae(args):
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae")
    vae.requires_grad_(False)
    vae.train()
    
    l_target_modules_encoder = []
    l_grep = ["conv1","conv2","conv_in", "conv_shortcut", "conv", "conv_out", "to_k", "to_q", "to_v", "to_out.0"]
    for n, p in vae.named_parameters():
        if "bias" in n or "norm" in n: 
            continue
        for pattern in l_grep:
            if pattern in n and ("encoder" in n):
                l_target_modules_encoder.append(n.replace(".weight",""))
            elif ('quant_conv' in n) and ('post_quant_conv' not in n):
                l_target_modules_encoder.append(n.replace(".weight",""))
    
    lora_conf_encoder = LoraConfig(r=args.lora_rank, init_lora_weights="gaussian",target_modules=l_target_modules_encoder)
    vae.add_adapter(lora_conf_encoder, adapter_name="default_encoder")

    return vae, l_target_modules_encoder


def initialize_unet(args, return_lora_module_names=False, pretrained_model_name_or_path=None):
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet")
    unet.requires_grad_(False)
    unet.train()

    l_target_modules_encoder, l_target_modules_decoder, l_modules_others = [], [], []
    l_grep = ["to_k", "to_q", "to_v", "to_out.0", "conv", "conv1", "conv2", "conv_in", "conv_shortcut", "conv_out", "proj_out", "proj_in", "ff.net.2", "ff.net.0.proj"]
    for n, p in unet.named_parameters():
        if "bias" in n or "norm" in n:
            continue
        for pattern in l_grep:
            if pattern in n and ("down_blocks" in n or "conv_in" in n):
                l_target_modules_encoder.append(n.replace(".weight",""))
                break
            elif pattern in n and ("up_blocks" in n or "conv_out" in n):
                l_target_modules_decoder.append(n.replace(".weight",""))
                break
            elif pattern in n:
                l_modules_others.append(n.replace(".weight",""))
                break

    lora_conf_encoder = LoraConfig(r=args.lora_rank, init_lora_weights="gaussian",target_modules=l_target_modules_encoder)
    lora_conf_decoder = LoraConfig(r=args.lora_rank, init_lora_weights="gaussian",target_modules=l_target_modules_decoder)
    lora_conf_others = LoraConfig(r=args.lora_rank, init_lora_weights="gaussian",target_modules=l_modules_others)
    unet.add_adapter(lora_conf_encoder, adapter_name="default_encoder")
    unet.add_adapter(lora_conf_decoder, adapter_name="default_decoder")
    unet.add_adapter(lora_conf_others, adapter_name="default_others")

    return unet, l_target_modules_encoder, l_target_modules_decoder, l_modules_others


class OSEDiff_gen(torch.nn.Module):
    def __init__(self, args):
        super().__init__()

        self.tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder").cuda()
        self.noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
        self.noise_scheduler.set_timesteps(1, device="cuda")
        self.noise_scheduler.alphas_cumprod = self.noise_scheduler.alphas_cumprod.cuda()
        self.args = args

        self.vae, self.lora_vae_modules_encoder = initialize_vae(self.args)
        self.unet, self.lora_unet_modules_encoder, self.lora_unet_modules_decoder, self.lora_unet_others = initialize_unet(self.args)
        self.lora_rank_unet = self.args.lora_rank
        self.lora_rank_vae = self.args.lora_rank

        self.unet.to("cuda")
        self.vae.to("cuda")
        self.timesteps = torch.tensor([999], device="cuda").long()
        self.text_encoder.requires_grad_(False)

    def set_train(self):
        self.unet.train()
        self.vae.train()
        for n, _p in self.unet.named_parameters():
            if "lora" in n:
                _p.requires_grad = True
        self.unet.conv_in.requires_grad_(True)
        for n, _p in self.vae.named_parameters():
            if "lora" in n:
                _p.requires_grad = True

    def encode_prompt(self, prompt_batch):
        prompt_embeds_list = []
        with torch.no_grad():
            for caption in prompt_batch:
                text_input_ids = self.tokenizer(
                    caption, max_length=self.tokenizer.model_max_length,
                    padding="max_length", truncation=True, return_tensors="pt"
                ).input_ids
                prompt_embeds = self.text_encoder(
                    text_input_ids.to(self.text_encoder.device),
                )[0]
                prompt_embeds_list.append(prompt_embeds)
        prompt_embeds = torch.concat(prompt_embeds_list, dim=0)
        return prompt_embeds

    def forward(self, c_t, batch=None, args=None):

        encoded_control = self.vae.encode(c_t).latent_dist.sample() * self.vae.config.scaling_factor

        if args.use_vre or args.ram_img_only:
            prompt_embeds = batch['v_prompt']
            # type(prompt_embeds)=<class 'torch.Tensor'>, prompt_embeds.shape=torch.Size([bs, 77, 1024])
            neg_prompt_embeds = torch.zeros_like(prompt_embeds)
        else:
            # calculate prompt_embeddings and neg_prompt_embeddings
            prompt_embeds = self.encode_prompt(batch["prompt"])
            # type(prompt_embeds)=<class 'torch.Tensor'>, prompt_embeds.shape=torch.Size([bs, 77, 1024])
            neg_prompt_embeds = self.encode_prompt(batch["neg_prompt"])
            # batch["neg_prompt"] is integraded in DataLoader

        model_pred = self.unet(encoded_control, self.timesteps, encoder_hidden_states=prompt_embeds.to(torch.float32),).sample
        x_denoised = self.noise_scheduler.step(model_pred, self.timesteps, encoded_control, return_dict=True).prev_sample
        output_image = (self.vae.decode(x_denoised / self.vae.config.scaling_factor).sample).clamp(-1, 1)

        return output_image, x_denoised, prompt_embeds, neg_prompt_embeds

    def save_model(self, outf):
        sd = {}
        sd["vae_lora_encoder_modules"] = self.lora_vae_modules_encoder
        sd["unet_lora_encoder_modules"], sd["unet_lora_decoder_modules"], sd["unet_lora_others_modules"] =\
            self.lora_unet_modules_encoder, self.lora_unet_modules_decoder, self.lora_unet_others
        sd["rank_unet"] = self.lora_rank_unet
        sd["rank_vae"] = self.lora_rank_vae
        sd["state_dict_unet"] = {k: v for k, v in self.unet.state_dict().items() if "lora" in k or "conv_in" in k}
        sd["state_dict_vae"] = {k: v for k, v in self.vae.state_dict().items() if "lora" in k}
        torch.save(sd, outf)

class OSEDiff_reg(torch.nn.Module):
    def __init__(self, args, accelerator):
        super().__init__() 

        self.tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder")
        self.noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
        self.args = args

        weight_dtype = torch.float32
        if accelerator.mixed_precision == "fp16":
            weight_dtype = torch.float16
        elif accelerator.mixed_precision == "bf16":
            weight_dtype = torch.bfloat16
        self.weight_dtype = weight_dtype

        self.vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae")
        self.unet_fix = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet")
        self.unet_update, self.lora_unet_modules_encoder, self.lora_unet_modules_decoder, self.lora_unet_others =\
                initialize_unet(args)

        self.text_encoder.to(accelerator.device, dtype=weight_dtype)
        self.unet_fix.to(accelerator.device, dtype=weight_dtype)
        self.unet_update.to(accelerator.device)
        self.vae.to(accelerator.device)
        
        self.text_encoder.requires_grad_(False)
        self.vae.requires_grad_(False)
        self.unet_fix.requires_grad_(False)

    def set_train(self):
        self.unet_update.train()
        for n, _p in self.unet_update.named_parameters():
            if "lora" in n:
                _p.requires_grad = True

    def diff_loss(self, latents, prompt_embeds, args):

        latents, prompt_embeds = latents.detach(), prompt_embeds.detach()
        noise = torch.randn_like(latents)
        bsz = latents.shape[0]
        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device).long()
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

        noise_pred = self.unet_update(
        noisy_latents,
        timestep=timesteps,
        encoder_hidden_states=prompt_embeds,
        ).sample

        loss_d = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")
        
        return loss_d

    def eps_to_mu(self, scheduler, model_output, sample, timesteps):
        
        alphas_cumprod = scheduler.alphas_cumprod.to(device=sample.device, dtype=sample.dtype)
        alpha_prod_t = alphas_cumprod[timesteps]
        while len(alpha_prod_t.shape) < len(sample.shape):
            alpha_prod_t = alpha_prod_t.unsqueeze(-1)
        beta_prod_t = 1 - alpha_prod_t
        pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
        return pred_original_sample

    def distribution_matching_loss(self, latents, prompt_embeds, neg_prompt_embeds, args):
        bsz = latents.shape[0]
        timesteps = torch.randint(20, 980, (bsz,), device=latents.device).long()
        noise = torch.randn_like(latents)
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

        with torch.no_grad():

            noise_pred_update = self.unet_update(
                noisy_latents,
                timestep=timesteps,
                encoder_hidden_states=prompt_embeds.float(),
                ).sample

            x0_pred_update = self.eps_to_mu(self.noise_scheduler, noise_pred_update, noisy_latents, timesteps)

            noisy_latents_input = torch.cat([noisy_latents] * 2)
            timesteps_input = torch.cat([timesteps] * 2)
            # prompt_embeds.shape=torch.Size([bs, 77, 1024]), neg_prompt_embeds.shape=torch.Size([bs, 77, 1024])
            prompt_embeds = torch.cat([neg_prompt_embeds, prompt_embeds], dim=0)

            noise_pred_fix = self.unet_fix(
                noisy_latents_input.to(dtype=self.weight_dtype),
                timestep=timesteps_input,
                encoder_hidden_states=prompt_embeds.to(dtype=self.weight_dtype),
                ).sample

            noise_pred_uncond, noise_pred_text = noise_pred_fix.chunk(2)
            noise_pred_fix = noise_pred_uncond + args.cfg_vsd * (noise_pred_text - noise_pred_uncond)
            noise_pred_fix.to(dtype=torch.float32)

            x0_pred_fix = self.eps_to_mu(self.noise_scheduler, noise_pred_fix, noisy_latents, timesteps)

        weighting_factor = torch.abs(latents - x0_pred_fix).mean(dim=[1, 2, 3], keepdim=True)

        grad = (x0_pred_update - x0_pred_fix) / weighting_factor
        loss = F.mse_loss(latents, (latents - grad).detach())

        return loss

class OSEDiff_test(torch.nn.Module):
    def __init__(self, args):
        super().__init__()

        self.args = args
        self.device =  torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.pretrained_model_name_or_path, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(self.args.pretrained_model_name_or_path, subfolder="text_encoder")
        self.noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
        self.noise_scheduler.set_timesteps(1, device="cuda")
        self.vae = AutoencoderKL.from_pretrained(self.args.pretrained_model_name_or_path, subfolder="vae")
        self.unet = UNet2DConditionModel.from_pretrained(self.args.pretrained_model_name_or_path, subfolder="unet")

        # vae tile
        self._init_tiled_vae(encoder_tile_size=args.vae_encoder_tiled_size, decoder_tile_size=args.vae_decoder_tiled_size)

        self.weight_dtype = torch.float32
        if args.mixed_precision == "fp16":
            self.weight_dtype = torch.float16


        # self.merge(osediff)
        if not getattr(args, 'zeroshot', False):
            osediff = torch.load(args.osediff_path)
            self.load_ckpt(osediff)

        # merge lora
            if self.args.merge_and_unload_lora:
                print(f'===> MERGE LORA <===')
                self.vae = self.vae.merge_and_unload()
                self.unet = self.unet.merge_and_unload()

        self.unet.to("cuda", dtype=self.weight_dtype)
        self.vae.to("cuda", dtype=self.weight_dtype)
        self.text_encoder.to("cuda", dtype=self.weight_dtype)
        self.timesteps = torch.tensor([getattr(args, 'timestep', 999)], device="cuda").long()
        self.noise_scheduler.alphas_cumprod = self.noise_scheduler.alphas_cumprod.cuda()

        

    def load_ckpt(self, model):
        # load unet lora
        lora_conf_encoder = LoraConfig(r=model["rank_unet"], init_lora_weights="gaussian", target_modules=model["unet_lora_encoder_modules"])
        lora_conf_decoder = LoraConfig(r=model["rank_unet"], init_lora_weights="gaussian", target_modules=model["unet_lora_decoder_modules"])
        lora_conf_others = LoraConfig(r=model["rank_unet"], init_lora_weights="gaussian", target_modules=model["unet_lora_others_modules"])
        self.unet.add_adapter(lora_conf_encoder, adapter_name="default_encoder")
        self.unet.add_adapter(lora_conf_decoder, adapter_name="default_decoder")
        self.unet.add_adapter(lora_conf_others, adapter_name="default_others")
        for n, p in self.unet.named_parameters():
            if "lora" in n or "conv_in" in n:
                p.data.copy_(model["state_dict_unet"][n])
        self.unet.set_adapter(["default_encoder", "default_decoder", "default_others"])

        # load vae lora
        vae_lora_conf_encoder = LoraConfig(r=model["rank_vae"], init_lora_weights="gaussian", target_modules=model["vae_lora_encoder_modules"])
        self.vae.add_adapter(vae_lora_conf_encoder, adapter_name="default_encoder")
        for n, p in self.vae.named_parameters():
            if "lora" in n:
                p.data.copy_(model["state_dict_vae"][n])
        self.vae.set_adapter(['default_encoder'])

    def merge(self,osediff):
        args = self.args
        unet = self.unet
        vae = self.vae
        lora_alpha = 5.5
        lora_rank = 16
        alpha = float(lora_alpha / lora_rank)
        processed_keys_vae = set()
        processed_keys_unet = set()
        state_dict_unet = unet.state_dict()
        state_dict_vae = vae.state_dict()
        state_dict_unet["conv_in.weight"] = osediff["state_dict_unet"]["conv_in.base_layer.weight"]
        state_dict_unet["conv_in.bias"] = osediff["state_dict_unet"]["conv_in.base_layer.bias"]
        for key in osediff["state_dict_unet"].keys():

            if "lora_A" in key:
                lora_a_key = key
                lora_b_key = key.replace("lora_A", "lora_B")
                unet_key = key.replace(".lora_A", "").replace("default_encoder.", "").replace("default_decoder.", "").replace("default_others.", "")

                assert lora_b_key in osediff["state_dict_unet"] and unet_key in state_dict_unet,f"{lora_b_key} and {unet_key}"
                original_weight = state_dict_unet[unet_key]
                W_A = osediff["state_dict_unet"][lora_a_key].to(original_weight)
                W_B = osediff["state_dict_unet"][lora_b_key].to(original_weight)
                processed_keys_unet.update([lora_a_key, lora_b_key])
                if len(original_weight.shape) == 4 and len(W_A.shape) == 4 and len(W_B.shape) == 4:
                    out_channels, in_channels, kH, kW = original_weight.shape
                    rank = W_A.shape[0]
                    # print(rank)
                    assert rank == lora_rank, f"lora rank should be {rank}, but {lora_alpha}"
                    assert W_A.shape == (rank, in_channels, kH, kW), "W_A has an unexpected shape"
                    assert W_B.shape == (out_channels, rank, 1, 1), "W_B has an unexpected shape"
                    W_A_flat = W_A.view(rank, -1)  # (rank, in_channels * kH * kW)
                    W_B_flat = W_B.view(out_channels, rank)  # (out_channels, rank)
                    delta_W_flat = torch.matmul(W_B_flat, W_A_flat)  # (out_channels, in_channels * kH * kW)
                    delta_W = delta_W_flat.view(out_channels, in_channels, kH, kW).to(original_weight)
                    merged_weight = original_weight + alpha * delta_W
                else:
                    merged_weight = original_weight + alpha * torch.mm(W_B, W_A)
                state_dict_unet[unet_key] = merged_weight
            elif 'lora.up.weight' in key:
                lora_up_key = key
                lora_down_key = key.replace('lora.up.weight', 'lora.down.weight')

                original_weight_key = key.replace('.lora.up.weight', '.weight').replace("unet.", "")
                assert lora_down_key in osediff["state_dict_unet"] and original_weight_key in state_dict_unet
                W_up = osediff["state_dict_unet"][lora_up_key]
                W_down = osediff["state_dict_unet"][lora_down_key]
                W_orig = state_dict_unet[original_weight_key]
                processed_keys_unet.update([lora_up_key, lora_down_key])

                if W_orig.ndim == 2:
                    delta_W = torch.matmul(W_up, W_down).to(W_orig)
                    W_merged = W_orig + alpha * delta_W

                else:
                    print(f"Warning: Unhandled weight shape for {original_weight_key}")
                    continue
                state_dict_unet[original_weight_key] = W_merged

        remaining_lora_keys = [k for k in osediff["state_dict_unet"].keys() if k not in processed_keys_unet]
        if remaining_lora_keys:
            print("Warning: There are unprocessed LoRA weights:")
            for key in remaining_lora_keys:
                print(f" - {key}")
        self.unet.load_state_dict(state_dict_unet)

        for key in osediff["state_dict_vae"].keys():
            if "lora_A" in key:
                lora_a_key = key
                lora_b_key = key.replace("lora_A", "lora_B")
                unet_key = (key.replace(".lora_A", "").replace("default_encoder.", "")
                            .replace("default_decoder.","").replace("default_others.", ""))

                assert lora_b_key in osediff["state_dict_vae"] and unet_key in state_dict_vae
                original_weight = state_dict_vae[unet_key]
                W_A = osediff["state_dict_vae"][lora_a_key].to(original_weight)
                W_B = osediff["state_dict_vae"][lora_b_key].to(original_weight)
                processed_keys_vae.update([lora_a_key, lora_b_key])
                if len(original_weight.shape) == 4 and len(W_A.shape) == 4 and len(W_B.shape) == 4:
                    out_channels, in_channels, kH, kW = original_weight.shape
                    rank = W_A.shape[0]
                    # print(rank)
                    assert rank == lora_rank, f"lora rank should be {rank}, but {lora_alpha}"
                    assert W_A.shape == (rank, in_channels, kH, kW), "W_A has an unexpected shape"
                    assert W_B.shape == (out_channels, rank, 1, 1), "W_B has an unexpected shape"
                    W_A_flat = W_A.view(rank, -1).to(original_weight)  # (rank, in_channels * kH * kW)
                    W_B_flat = W_B.view(out_channels, rank).to(original_weight)  # (out_channels, rank)
                    delta_W_flat = torch.matmul(W_B_flat, W_A_flat)  # (out_channels, in_channels * kH * kW)
                    delta_W = delta_W_flat.view(out_channels, in_channels, kH, kW).to(original_weight)
                    merged_weight = original_weight + alpha * delta_W
                else:
                    merged_weight = original_weight + alpha * torch.mm(W_B, W_A)
                state_dict_vae[unet_key] = merged_weight
            elif 'lora.up.weight' in key:
                lora_up_key = key
                lora_down_key = key.replace('lora.up.weight', 'lora.down.weight')

                original_weight_key = key.replace('.lora.up.weight', '.weight').replace("unet.", "")
                assert lora_down_key in osediff["state_dict_vae"] and original_weight_key in state_dict_vae
                W_up = osediff["state_dict_vae"][lora_up_key]
                W_down = osediff["state_dict_vae"][lora_down_key]
                W_orig = state_dict_vae[original_weight_key]
                processed_keys_vae.update([lora_up_key, lora_down_key])

                if W_orig.ndim == 2:
                    delta_W = torch.matmul(W_up, W_down).to(W_orig)
                    W_merged = W_orig + alpha * delta_W

                else:
                    print(f"Warning: Unhandled weight shape for {original_weight_key}")
                    continue
                state_dict_vae[original_weight_key] = W_merged

        remaining_lora_keys = [k for k in osediff["state_dict_vae"].keys() if k not in processed_keys_vae]
        if remaining_lora_keys:
            print("Warning: There are unprocessed LoRA weights:")
            for key in remaining_lora_keys:
                print(f" - {key}")
        print("Merge Done!")
        self.vae.load_state_dict(state_dict_vae)

    def encode_prompt(self, prompt_batch):
        prompt_embeds_list = []
        with torch.no_grad():
            for caption in prompt_batch:
                text_input_ids = self.tokenizer(
                    caption, max_length=self.tokenizer.model_max_length,
                    padding="max_length", truncation=True, return_tensors="pt"
                ).input_ids
                prompt_embeds = self.text_encoder(
                    text_input_ids.to(self.text_encoder.device),
                )[0]
                prompt_embeds_list.append(prompt_embeds)
        prompt_embeds = torch.concat(prompt_embeds_list, dim=0)
        return prompt_embeds

    # @perfcount
    @torch.no_grad()
    def forward(self, lq, prompt=None, visual_prompt=None):

        if self.args.use_vre or self.args.ram_img_only:
            prompt_embeds = visual_prompt
        else:
            prompt_embeds = self.encode_prompt([prompt])

        lq_latent = self.vae.encode(lq.to(self.weight_dtype)).latent_dist.sample() * self.vae.config.scaling_factor
        # save_tensor_as_images(lq_latent, "lq_latent")
        ## add tile function
        _, _, h, w = lq_latent.size()
        tile_size, tile_overlap = (self.args.latent_tiled_size, self.args.latent_tiled_overlap)
        if h * w <= tile_size * tile_size:
            print(f"[Tiled Latent]: the input size is tiny and unnecessary to tile.")
            model_pred = self.unet(lq_latent, self.timesteps, encoder_hidden_states=prompt_embeds).sample
            # save_tensor_as_images(model_pred, "model_pred")
        else:
            print(f"[Tiled Latent]: the input size is {lq.shape[-2]}x{lq.shape[-1]}, need to tiled")
            tile_weights = self._gaussian_weights(tile_size, tile_size, 1)
            tile_size = min(tile_size, min(h, w))
            tile_weights = self._gaussian_weights(tile_size, tile_size, 1)

            grid_rows = 0
            cur_x = 0
            while cur_x < lq_latent.size(-1):
                cur_x = max(grid_rows * tile_size-tile_overlap * grid_rows, 0)+tile_size
                grid_rows += 1

            grid_cols = 0
            cur_y = 0
            while cur_y < lq_latent.size(-2):
                cur_y = max(grid_cols * tile_size-tile_overlap * grid_cols, 0)+tile_size
                grid_cols += 1

            input_list = []
            noise_preds = []
            for row in range(grid_rows):
                noise_preds_row = []
                for col in range(grid_cols):
                    if col < grid_cols-1 or row < grid_rows-1:
                        # extract tile from input image
                        ofs_x = max(row * tile_size-tile_overlap * row, 0)
                        ofs_y = max(col * tile_size-tile_overlap * col, 0)
                        # input tile area on total image
                    if row == grid_rows-1:
                        ofs_x = w - tile_size
                    if col == grid_cols-1:
                        ofs_y = h - tile_size

                    input_start_x = ofs_x
                    input_end_x = ofs_x + tile_size
                    input_start_y = ofs_y
                    input_end_y = ofs_y + tile_size

                    # input tile dimensions
                    input_tile = lq_latent[:, :, input_start_y:input_end_y, input_start_x:input_end_x]
                    input_list.append(input_tile)

                    if len(input_list) == 1 or col == grid_cols-1:
                        input_list_t = torch.cat(input_list, dim=0)
                        # predict the noise residual
                        model_out = self.unet(input_list_t, self.timesteps, encoder_hidden_states=prompt_embeds.to(self.weight_dtype),).sample
                        input_list = []
                    noise_preds.append(model_out)

            # Stitch noise predictions for all tiles
            noise_pred = torch.zeros(lq_latent.shape, device=lq_latent.device)
            contributors = torch.zeros(lq_latent.shape, device=lq_latent.device)
            # Add each tile contribution to overall latents
            for row in range(grid_rows):
                for col in range(grid_cols):
                    if col < grid_cols-1 or row < grid_rows-1:
                        # extract tile from input image
                        ofs_x = max(row * tile_size-tile_overlap * row, 0)
                        ofs_y = max(col * tile_size-tile_overlap * col, 0)
                        # input tile area on total image
                    if row == grid_rows-1:
                        ofs_x = w - tile_size
                    if col == grid_cols-1:
                        ofs_y = h - tile_size

                    input_start_x = ofs_x
                    input_end_x = ofs_x + tile_size
                    input_start_y = ofs_y
                    input_end_y = ofs_y + tile_size

                    noise_pred[:, :, input_start_y:input_end_y, input_start_x:input_end_x] += noise_preds[row*grid_cols + col] * tile_weights
                    contributors[:, :, input_start_y:input_end_y, input_start_x:input_end_x] += tile_weights
            # Average overlapping areas with more than 1 contributor
            noise_pred /= contributors
            model_pred = noise_pred

        x_denoised = self.noise_scheduler.step(model_pred, self.timesteps, lq_latent, return_dict=True).prev_sample
        output_image = (self.vae.decode(x_denoised.to(self.weight_dtype) / self.vae.config.scaling_factor).sample).clamp(-1, 1)

        return output_image

    def _init_tiled_vae(self,
            encoder_tile_size = 256,
            decoder_tile_size = 256,
            fast_decoder = False,
            fast_encoder = False,
            color_fix = False,
            vae_to_gpu = True):
        # save original forward (only once)
        if not hasattr(self.vae.encoder, 'original_forward'):
            setattr(self.vae.encoder, 'original_forward', self.vae.encoder.forward)
        if not hasattr(self.vae.decoder, 'original_forward'):
            setattr(self.vae.decoder, 'original_forward', self.vae.decoder.forward)

        encoder = self.vae.encoder
        decoder = self.vae.decoder

        self.vae.encoder.forward = VAEHook(
            encoder, encoder_tile_size, is_decoder=False, fast_decoder=fast_decoder, fast_encoder=fast_encoder, color_fix=color_fix, to_gpu=vae_to_gpu)
        self.vae.decoder.forward = VAEHook(
            decoder, decoder_tile_size, is_decoder=True, fast_decoder=fast_decoder, fast_encoder=fast_encoder, color_fix=color_fix, to_gpu=vae_to_gpu)

    def _gaussian_weights(self, tile_width, tile_height, nbatches):
        """Generates a gaussian mask of weights for tile contributions"""
        from numpy import pi, exp, sqrt
        import numpy as np

        latent_width = tile_width
        latent_height = tile_height

        var = 0.01
        midpoint = (latent_width - 1) / 2  # -1 because index goes from 0 to latent_width - 1
        x_probs = [exp(-(x-midpoint)*(x-midpoint)/(latent_width*latent_width)/(2*var)) / sqrt(2*pi*var) for x in range(latent_width)]
        midpoint = latent_height / 2
        y_probs = [exp(-(y-midpoint)*(y-midpoint)/(latent_height*latent_height)/(2*var)) / sqrt(2*pi*var) for y in range(latent_height)]

        weights = np.outer(y_probs, x_probs)
        return torch.tile(torch.tensor(weights, device=self.device), (nbatches, self.unet.config.in_channels, 1, 1))


class OSEDiff_inference_time(torch.nn.Module):
    def __init__(self, args):
        super().__init__()

        self.args = args
        self.device =  torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.pretrained_model_name_or_path, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(self.args.pretrained_model_name_or_path, subfolder="text_encoder")
        self.noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
        self.noise_scheduler.set_timesteps(1, device="cuda")
        self.vae = AutoencoderKL.from_pretrained(self.args.pretrained_model_name_or_path, subfolder="vae")
        self.unet = UNet2DConditionModel.from_pretrained(self.args.pretrained_model_name_or_path, subfolder="unet")

        self.weight_dtype = torch.float32
        if args.mixed_precision == "fp16":
            self.weight_dtype = torch.float16

        osediff = torch.load(args.osediff_path)
        self.load_ckpt(osediff)

        # merge lora
        if self.args.merge_and_unload_lora:
            print(f'===> MERGE LORA <===')
            self.vae = self.vae.merge_and_unload()
            self.unet = self.unet.merge_and_unload()

        self.unet.to("cuda", dtype=self.weight_dtype)
        self.vae.to("cuda", dtype=self.weight_dtype)
        self.text_encoder.to("cuda", dtype=self.weight_dtype)
        self.timesteps = torch.tensor([999], device="cuda").long() 
        self.noise_scheduler.alphas_cumprod = self.noise_scheduler.alphas_cumprod.cuda()

        

    def load_ckpt(self, model):
        # load unet lora
        lora_conf_encoder = LoraConfig(r=model["rank_unet"], init_lora_weights="gaussian", target_modules=model["unet_lora_encoder_modules"])
        lora_conf_decoder = LoraConfig(r=model["rank_unet"], init_lora_weights="gaussian", target_modules=model["unet_lora_decoder_modules"])
        lora_conf_others = LoraConfig(r=model["rank_unet"], init_lora_weights="gaussian", target_modules=model["unet_lora_others_modules"])
        self.unet.add_adapter(lora_conf_encoder, adapter_name="default_encoder")
        self.unet.add_adapter(lora_conf_decoder, adapter_name="default_decoder")
        self.unet.add_adapter(lora_conf_others, adapter_name="default_others")
        for n, p in self.unet.named_parameters():
            if "lora" in n or "conv_in" in n:
                p.data.copy_(model["state_dict_unet"][n])
        self.unet.set_adapter(["default_encoder", "default_decoder", "default_others"])

        # load vae lora
        vae_lora_conf_encoder = LoraConfig(r=model["rank_vae"], init_lora_weights="gaussian", target_modules=model["vae_lora_encoder_modules"])
        self.vae.add_adapter(vae_lora_conf_encoder, adapter_name="default_encoder")
        for n, p in self.vae.named_parameters():
            if "lora" in n:
                p.data.copy_(model["state_dict_vae"][n])
        self.vae.set_adapter(['default_encoder'])

    def encode_prompt(self, prompt_batch):
        prompt_embeds_list = []
        with torch.no_grad():
            for caption in prompt_batch:
                text_input_ids = self.tokenizer(
                    caption, max_length=self.tokenizer.model_max_length,
                    padding="max_length", truncation=True, return_tensors="pt"
                ).input_ids
                prompt_embeds = self.text_encoder(
                    text_input_ids.to(self.text_encoder.device),
                )[0]
                prompt_embeds_list.append(prompt_embeds)
        prompt_embeds = torch.concat(prompt_embeds_list, dim=0)
        return prompt_embeds

    @torch.no_grad()
    def forward(self, lq, prompt=None, visual_prompt=None):
        if self.args.use_vre or self.args.ram_img_only:
            prompt_embeds = visual_prompt
        else:
            prompt_embeds = self.encode_prompt([prompt])

        lq_latent = self.vae.encode(lq.to(self.weight_dtype)).latent_dist.sample() * self.vae.config.scaling_factor
        model_pred = self.unet(lq_latent, self.timesteps, encoder_hidden_states=prompt_embeds).sample
        x_denoised = self.noise_scheduler.step(model_pred, self.timesteps, lq_latent, return_dict=True).prev_sample
        output_image = (self.vae.decode(x_denoised.to(self.weight_dtype) / self.vae.config.scaling_factor).sample).clamp(-1, 1)

        return output_image
