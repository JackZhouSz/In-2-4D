"""Stage 1 — motion prompt from the two states with Qwen2.5-VL.

Output: video/caption/caption.json {"prompt", "negative_prompt", "model"}
"""
import torch
from PIL import Image

from core.io import composite_white, read_image, write_json
from core.paths import WAN, use_repo
from core.stage import StageContext, stage_parser

SYSTEM = "You write prompts for an image-to-video model."
INSTRUCTION = (
    "The first image is the START state and the second is the END state of the same object.\n"
    "Describe in one or two sentences (max 60 words) the physical motion that takes the object\n"
    "from START to END. Name the object, describe the motion and its direction, and state that\n"
    "the camera is static and the background is plain white. Do not describe the images separately."
)


def caption_two_images(model_name, start: Image.Image, end: Image.Image, max_new_tokens: int,
                       dtype="bfloat16", device_map="cuda") -> str:
    """Two-image variant of Wan2.2/wan/utils/prompt_extend.py::QwenPromptExpander.extend_with_img."""
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    try:
        from wan.utils.qwen_vl_utils import process_vision_info
    except ImportError:
        from qwen_vl_utils import process_vision_info

    processor = AutoProcessor.from_pretrained(model_name, min_pixels=256 * 28 * 28, max_pixels=1280 * 28 * 28,
                                              use_fast=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_name, torch_dtype=getattr(torch, dtype),
                                                               device_map=device_map).eval()
    messages = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM}]},
        {"role": "user", "content": [{"type": "image", "image": start}, {"type": "image", "image": end},
                                     {"type": "text", "text": INSTRUCTION}]},
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True,
                       return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, out)]
    return processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()


def negative_prompt(cfg) -> str:
    from wan.configs.shared_config import wan_shared_cfg

    return f"{wan_shared_cfg.sample_neg_prompt}, {cfg.caption.extra_negative}"


def main():
    p = stage_parser(__doc__)
    p.add_argument("--prompt", default=None, help="use this motion prompt instead of captioning")
    ctx = StageContext(p.parse_args(), 1)
    out = ctx.out / "caption.json"
    if ctx.done(out):
        return
    use_repo(WAN)

    prompt = ctx.args.prompt or ctx.cfg.caption.prompt
    model = "user" if prompt else ctx.cfg.caption.model
    if not prompt:
        start = Image.fromarray(composite_white(read_image(ctx.dirs.s0 / "start.png", "RGBA")))
        end = Image.fromarray(composite_white(read_image(ctx.dirs.s0 / "end.png", "RGBA")))
        c = ctx.cfg.caption
        prompt = caption_two_images(c.model, start, end, c.max_new_tokens, c.dtype, c.device_map)
    write_json({"prompt": prompt, "negative_prompt": negative_prompt(ctx.cfg), "model": model}, out)
    print(f"[s1] prompt: {prompt}")


if __name__ == "__main__":
    main()
