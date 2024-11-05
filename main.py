from utils.data_processing import get_image_paths
from utils.clip_utils import load_clip_model, generate_clip_prompt
from utils.blip_utils import load_blip_model, generate_blip_prompt
import os

def main():
    clip_model, clip_processor = load_clip_model()
    
    blip_model, blip_processor = load_blip_model()
    
    image_dir = "data/images/"
    
    image_paths = get_image_paths(image_dir)
    
    for image_path in image_paths:
        clip_prompt = generate_clip_prompt(clip_model, clip_processor, image_path)
        print(f"CLIP Generated Prompt: {clip_prompt}")
        
        blip_prompt = generate_blip_prompt(blip_model, blip_processor, image_path)
        print(f"BLIP Generated Prompt: {blip_prompt}")


if __name__ == "__main__":
    main()


