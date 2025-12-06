import torch
from diffusers import StableDiffusionXLPipeline

DIR = "nupzuki"

def main():
    model_id = "stabilityai/stable-diffusion-xl-base-1.0"

    pipe = StableDiffusionXLPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        use_safetensors=True,
    )
    pipe = pipe.to("cuda")

    prompt = "blue character, long width head, tiny body, big head, simple geometry, simple black eyes, simple background, totally blue, digital art, two legs, two arms, cute, two eyes, no mouth, simple eyes, short height head"
    negative_prompt = "low quality, blurry, distorted, ugly, text, watermark"

    '''
    with open(f"{DIR}/seeds.txt", "r") as f:
        existing_seeds = f.readlines()
        line_cnt = len(existing_seeds)
    '''

    for i in range(1):
        # 1) set seed
        # seed = torch.randint(0, 2**32, (1,)).item()
        seed = 641892446

        # 2) apply seed to generator
        generator = torch.Generator(device="cuda").manual_seed(seed)

        # 3) generate image
        image = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=30,
            guidance_scale=7.0,
            generator=generator,
        ).images[0]

        out_path = f"{DIR}/nupzuki.png"
        image.save(out_path)

        # 4) print seed
        print(f"saved to {out_path}, prompt={prompt}, negative_prompt={negative_prompt}, seed={seed}")
        with open(f"{DIR}/seeds.txt", "a") as f:
            f.write(f"saved to {out_path}, prompt={prompt}, negative_prompt={negative_prompt}, seed={seed}\n")

if __name__ == "__main__":
    main()
