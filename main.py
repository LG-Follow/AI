from flask import Flask, request, jsonify
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import requests
from io import BytesIO

app = Flask(__name__)

# 모델과 프로세서 로드
model_path = "project_directory/models/trained_model"
processor = BlipProcessor.from_pretrained(model_path)
model = BlipForConditionalGeneration.from_pretrained(model_path)
model.to("cuda" if torch.cuda.is_available() else "cpu")

@app.route("/generate-prompt", methods=["POST"])
def generate_prompt():
    data = request.get_json()
    image_url = data.get("image_url")
    if not image_url:
        return jsonify({"error": "No image URL provided"}), 400

    # S3 URL에서 이미지 가져오기
    response = requests.get(image_url)
    image = Image.open(BytesIO(response.content)).convert("RGB")

    # 프롬프트 생성
    inputs = processor(images=image, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=50)
    generated_prompt = processor.decode(output[0], skip_special_tokens=True)

    return jsonify({"generated_prompt": generated_prompt})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
