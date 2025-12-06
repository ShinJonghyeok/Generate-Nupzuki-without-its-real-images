import os
import torch
from diffusers import StableDiffusionXLImg2ImgPipeline
from PIL import Image

model_id = "stabilityai/stable-diffusion-xl-base-1.0"

pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16",
).to("cuda")

# LoRA list
lora_list = [
    "TheLastBen/Papercut_SDXL",
    "ostris/crayon_style_lora_sdxl",
    "joachimsallstrom/aether-cloud-lora-for-sdxl",
    "ostris/watercolor_style_lora_sdxl",
]

# lora scale for each lora
lora_scale_list = [0.2, 0.4, 0.2, 0.4]

seed_list = [1879173339, 333462361, 1778945006, 930866658]

prompt_list = [
    "make this character look like papercraft model. This character is non-human, non-animal and blue in color",
    "make this character look like crayon drawing. This character is non-human, non-animal and blue in color",
    "make this character look like aether cloud style. This character is non-human, non-animal and blue in color",
    "make this character look like watercolor painting. This character is non-human, non-animal and blue in color",
]

prompt = ""
negative_prompt = "low quality, blurry, distorted, human, equipment, things"

init_image = Image.open("./nupzuki/nupzuki.png").convert("RGB")
init_image = init_image.resize((1024, 1024))

STRENGTH = 1.0
NUM_INFERENCE_STEPS = 30
GUIDANCE_SCALE = 7.5

for i, (lora_id, prompt, lora_scale, seed) in enumerate(zip(lora_list, prompt_list, lora_scale_list, seed_list)):
    save_dir = os.path.join("./final")
    os.makedirs(save_dir, exist_ok=True)

    print(f"\n=== Loading LoRA: {lora_id} ===")

    # unload previous LoRA
    if hasattr(pipe, "unload_lora_weights"):
        try:
            pipe.unload_lora_weights()
        except Exception:
            pass
    if hasattr(pipe, "unfuse_lora"):
        try:
            pipe.unfuse_lora()
        except Exception:
            pass

    # Load LoRA
    pipe.load_lora_weights(
        lora_id,
        weight_name=None,
    )

    generator = torch.Generator(device="cuda").manual_seed(seed)

    image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=init_image,
        strength=STRENGTH,
        num_inference_steps=NUM_INFERENCE_STEPS,
        guidance_scale=GUIDANCE_SCALE,
        generator=generator,
        cross_attention_kwargs={"scale": lora_scale},
    ).images[0]

    filename = f"{i}.png"

    save_path = os.path.join(save_dir, filename)
    image.save(save_path)

    print(f"    Saved: {save_path}")
