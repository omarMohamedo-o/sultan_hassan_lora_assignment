import os
import torch
import shutil
import subprocess
import argparse
from diffusers import StableDiffusionXLPipeline, AutoencoderKL

def prepare_and_export_onnx(lora_path, output_onnx_dir):
    fused_dir = "temp_fused_sdxl"
    
    print("1. Loading base SDXL model...")
    # We use fp32 for ONNX export to ensure the graph traces perfectly without underflow
    vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float32)
    pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        vae=vae,
        torch_dtype=torch.float32
    )

    print(f"2. Loading and fusing LoRA weights from '{lora_path}'...")
    pipe.load_lora_weights(lora_path)
    # Fusing the LoRA bakes the math directly into the base model's UNet.
    # This is required because ONNX does not support dynamically loading LoRAs!
    pipe.fuse_lora()

    print(f"3. Saving fused PyTorch model temporarily to '{fused_dir}'...")
    pipe.save_pretrained(fused_dir)
    print("Temporary model saved.")

    print(f"4. Calling optimum-cli to convert the fused model to ONNX...")
    print("This will take a significant amount of RAM and time (10-20 minutes).")
    
    # Optimum CLI is the safest and most robust way to trace SDXL into ONNX graphs
    try:
        subprocess.run([
            "optimum-cli", "export", "onnx", 
            "--model", fused_dir, 
            "--task", "stable-diffusion-xl", 
            output_onnx_dir
        ], check=True)
        print(f"Success! Your ONNX model is fully built and saved in: {output_onnx_dir}")
    except subprocess.CalledProcessError as e:
        print(f"ONNX export failed: {e}")
    finally:
        print("Cleaning up temporary PyTorch files...")
        if os.path.exists(fused_dir):
            shutil.rmtree(fused_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fuse LoRA and Export SDXL to ONNX")
    parser.add_argument("--lora_path", type=str, default="sultan_hassan_sdxl_lora", help="Path to the trained LoRA folder")
    parser.add_argument("--output_dir", type=str, default="sultan_hassan_onnx", help="Where to save the final ONNX model")
    
    args = parser.parse_args()
    
    prepare_and_export_onnx(args.lora_path, args.output_dir)
