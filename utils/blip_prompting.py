import json
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image


# 설정 파일 로드
with open("configs/blip_config.json", "r") as f:
    config = json.load(f)

# config에서 model_dir을 가져옵니다.
model_dir = config["model_dir"]

def load_trained_model(model_dir):
    processor = BlipProcessor.from_pretrained(model_dir)
    model = BlipForConditionalGeneration.from_pretrained(model_dir)
    model.to("cuda")
    model.eval()
    return model, processor

def generate_prompt_from_image(image_path, model, processor):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to("cuda")
    output_ids = model.generate(**inputs, max_new_tokens=30)
    prompt = processor.decode(output_ids[0], skip_special_tokens=True)
    return prompt
