import torch
from diffusers import DiffusionPipeline, AutoencoderKL
import argparse
import os

def generate_image(prompt, lora_path, output_path="output.png"):
    print("Loading SDXL model and fp16-fix VAE...")
    
    # We use the fp16-fix VAE to prevent black images, just like in the notebook
    vae = AutoencoderKL.from_pretrained(
        "madebyollin/sdxl-vae-fp16-fix", 
        torch_dtype=torch.float16
    )
    
    # Load the base SDXL model
    pipe = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        vae=vae,
        torch_dtype=torch.float16,
        variant="fp16",
    ).to("cuda")

    print(f"Loading Sultan Hassan LoRA from {lora_path}...")
    pipe.load_lora_weights(lora_path)

    print(f"Generating image for prompt: '{prompt}'")
    # Generate the image
    image = pipe(prompt, num_inference_steps=30).images[0]

    # Save the image locally
    image.save(output_path)
    print(f"Success! Image saved to {os.path.abspath(output_path)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Sultan Hassan SDXL LoRA Locally")
    parser.add_argument("--lora_path", type=str, default="sultan_hassan_sdxl_lora", help="Path to your unzipped LoRA folder")
    parser.add_argument("--prompt", type=str, default="A beautiful professional photograph of sltnhsn at sunset, highly detailed, dramatic lighting", help="The prompt to generate")
    parser.add_argument("--output", type=str, default="sultan_hassan_local_test.png", help="Output file name")
    
    args = parser.parse_args()
    
    generate_image(args.prompt, args.lora_path, args.output)
