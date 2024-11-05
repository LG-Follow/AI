from transformers import BlipProcessor, BlipForConditionalGeneration
import torch
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_blip_model():
    """BLIP 모델 및 프로세서 로드"""
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)
    return model, processor

def generate_blip_prompt(model, processor, image_path, max_new_tokens=300):
    """BLIP 모델을 사용하여 이미지에서 서술적 프롬프트 생성"""
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"Error opening image: {e}")
        return None

    inputs = processor(images=image, return_tensors="pt").to(device)

    # BLIP 모델을 사용하여 프롬프트 생성
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)

    # 결과 디코딩 및 정제
    prompt = processor.decode(outputs[0], skip_special_tokens=True)
    return prompt
