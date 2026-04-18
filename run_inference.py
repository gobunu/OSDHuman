#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import random
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = SCRIPT_DIR / "test_set"
DEFAULT_OUTPUT = SCRIPT_DIR / "outputs"
DEFAULT_WEIGHTS = SCRIPT_DIR / "weights"
DEFAULT_PRESET = SCRIPT_DIR / "preset"
SUPPORTED_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run OSDHuman HFIE inference on one image or a directory of images."
    )
    parser.add_argument(
        "--input_image",
        "-i",
        type=str,
        default=str(DEFAULT_INPUT),
        help="Input image file or directory.",
    )
    parser.add_argument(
        "--output_dir",
        "-o",
        type=str,
        default=str(DEFAULT_OUTPUT),
        help="Directory used to save restored images.",
    )
    parser.add_argument(
        "--osediff_path",
        type=str,
        default=str(DEFAULT_WEIGHTS / "model.pkl"),
        help="Path to the OSDHuman LoRA checkpoint.",
    )
    parser.add_argument(
        "--embedding_change_path",
        type=str,
        default=str(DEFAULT_WEIGHTS / "model_embedding_change.pkl"),
        help="Path to the HFIE embedding adaptor checkpoint.",
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=str(DEFAULT_PRESET / "stable-diffusion-2-1-base"),
        help="Path to the Stable Diffusion 2.1 base model directory.",
    )
    parser.add_argument(
        "--ram_path",
        type=str,
        default=str(DEFAULT_PRESET / "ram_swin_large_14m.pth"),
        help="Path to the RAM backbone checkpoint.",
    )
    parser.add_argument(
        "--ram_ft_path",
        type=str,
        default=None,
        help="Optional RAM fine-tuned checkpoint used by prompt-based inference.",
    )
    parser.add_argument("--prompt", type=str, default="", help="Optional user prompt suffix.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--gpu_id", type=int, default=0, help="CUDA device index.")
    parser.add_argument("--process_size", type=int, default=512)
    parser.add_argument("--upscale", type=int, default=1)
    parser.add_argument(
        "--align_method",
        type=str,
        choices=["wavelet", "adain", "nofix"],
        default="adain",
        help="Color alignment applied after restoration.",
    )
    parser.add_argument("--mixed_precision", type=str, choices=["fp16", "fp32"], default="fp16")
    parser.add_argument(
        "--merge_and_unload_lora",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Merge LoRA weights into the base model before inference.",
    )
    parser.add_argument(
        "--use_vre",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use the alternate conditioning path.",
    )
    parser.add_argument(
        "--ram_img_only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use the HFIE image-embedding path instead of text prompts.",
    )
    parser.add_argument("--vae_decoder_tiled_size", type=int, default=224)
    parser.add_argument("--vae_encoder_tiled_size", type=int, default=1024)
    parser.add_argument("--latent_tiled_size", type=int, default=96)
    parser.add_argument("--latent_tiled_overlap", type=int, default=32)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def collect_image_paths(input_path: Path) -> list[Path]:
    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    if input_path.is_file():
        if input_path.suffix.lower() not in SUPPORTED_SUFFIXES:
            raise ValueError(f"Unsupported input file suffix: {input_path.suffix}")
        return [input_path]

    image_paths = sorted(
        path
        for path in input_path.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
    )
    if not image_paths:
        raise RuntimeError(f"No supported images found in {input_path}")
    return image_paths


def ensure_required_paths(args: argparse.Namespace) -> None:
    required = {
        "input_image": Path(args.input_image),
        "osediff_path": Path(args.osediff_path),
        "pretrained_model_name_or_path": Path(args.pretrained_model_name_or_path),
        "ram_path": Path(args.ram_path),
    }
    if args.ram_img_only or args.use_vre:
        required["embedding_change_path"] = Path(args.embedding_change_path)

    missing = [f"{name}: {path}" for name, path in required.items() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required paths:\n" + "\n".join(missing))


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    ensure_required_paths(args)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    import numpy as np
    import torch
    from PIL import Image
    from torchvision import transforms
    import torchvision.transforms.functional as tvf
    from tqdm import tqdm

    from my_utils.wavelet_color_fix import adain_color_fix, wavelet_color_fix
    from osediff import OSEDiff_test
    from ram import inference_ram as inference_ram_caption
    from ram import inference_ram_img_embed
    from ram.inference import AttentionPooling
    from ram.models.ram_lora import ram

    if not torch.cuda.is_available():
        raise RuntimeError("OSDHuman inference currently requires at least one CUDA device.")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    weight_dtype = torch.float16 if args.mixed_precision == "fp16" else torch.float32
    image_paths = collect_image_paths(Path(args.input_image))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tensor_transforms = transforms.Compose([transforms.ToTensor()])
    ram_transforms = transforms.Compose(
        [
            transforms.Resize((384, 384)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    model = OSEDiff_test(args)

    if args.ram_img_only or args.use_vre:
        ram_model = ram(
            ram_img_only=True,
            pretrained=args.ram_path,
            pretrained_condition=None,
            image_size=384,
            vit="swin_l",
        )
        embedding_change = AttentionPooling(args, 512, 77, 2)
        embedding_change.load_state_dict(torch.load(args.embedding_change_path, map_location="cpu"))
        embedding_change.eval().to("cuda", dtype=weight_dtype)
    else:
        ram_model = ram(
            pretrained=args.ram_path,
            pretrained_condition=args.ram_ft_path,
            image_size=384,
            vit="swin_l",
        )
        embedding_change = None

    ram_model.eval().to("cuda", dtype=weight_dtype)

    image_times: list[float] = []
    full_times: list[float] = []

    for image_path in tqdm(image_paths, desc="Running OSDHuman", dynamic_ncols=True):
        input_image = Image.open(image_path).convert("RGB")
        ori_width, ori_height = input_image.size
        resize_flag = False

        if ori_width < args.process_size // args.upscale or ori_height < args.process_size // args.upscale:
            scale = (args.process_size // args.upscale) / min(ori_width, ori_height)
            input_image = input_image.resize((int(scale * ori_width), int(scale * ori_height)))
            resize_flag = True

        input_image = input_image.resize(
            (input_image.size[0] * args.upscale, input_image.size[1] * args.upscale)
        )
        new_width = input_image.width - input_image.width % 8
        new_height = input_image.height - input_image.height % 8
        input_image = input_image.resize((new_width, new_height), Image.LANCZOS)

        current_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)
        full_start = torch.cuda.Event(enable_timing=True)
        full_end = torch.cuda.Event(enable_timing=True)

        with torch.no_grad():
            full_start.record()
            lq = tvf.to_tensor(input_image).unsqueeze(0).cuda() * 2 - 1
            lq = lq.to(dtype=weight_dtype)

            prompt = None
            visual_prompt = None
            if args.ram_img_only or args.use_vre:
                prompt_embedding = inference_ram_img_embed(ram_transforms(lq), ram_model)
                visual_prompt = embedding_change(prompt_embedding)
            else:
                ram_input = ram_transforms(tensor_transforms(input_image).unsqueeze(0).cuda())
                captions = inference_ram_caption(ram_input.to(dtype=weight_dtype), ram_model)
                prompt = f"{captions[0]}, {args.prompt}," if args.prompt else captions[0]

            current_time.record()
            output_image = model(lq, prompt=prompt, visual_prompt=visual_prompt)
            end_time.record()
            full_end.record()
            torch.cuda.synchronize()

            image_times.append(current_time.elapsed_time(end_time) / 1000.0)
            full_times.append(full_start.elapsed_time(full_end) / 1000.0)

            output_pil = transforms.ToPILImage()(output_image[0].cpu() * 0.5 + 0.5)
            if args.align_method == "adain":
                output_pil = adain_color_fix(target=output_pil, source=input_image)
            elif args.align_method == "wavelet":
                output_pil = wavelet_color_fix(target=output_pil, source=input_image)

            if resize_flag:
                output_pil = output_pil.resize(
                    (int(args.upscale * ori_width), int(args.upscale * ori_height)),
                    Image.LANCZOS,
                )

        output_pil.save(output_dir / image_path.name)

    if image_times:
        print(f"Processed {len(image_times)} image(s).")
        print(f"Average model time: {sum(image_times) / len(image_times):.4f}s")
        print(f"Average end-to-end time: {sum(full_times) / len(full_times):.4f}s")


if __name__ == "__main__":
    main()
