from flask import Flask, request, jsonify
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import requests
from io import BytesIO
from kafka import KafkaConsumer, KafkaProducer
import json
from datetime import datetime

app = Flask(__name__)

# 모델과 프로세서 로드
model_path = "project_directory/models/trained_model"
processor = BlipProcessor.from_pretrained(model_path)
model = BlipForConditionalGeneration.from_pretrained(model_path)
model.to("cuda" if torch.cuda.is_available() else "cpu")

# Kafka 설정
KAFKA_BROKER = "127.0.0.1:9092"  # Kafka 브로커 주소
IMAGE_TOPIC = "image-topic"      # Spring에서 보낸 이미지 URL을 수신할 토픽
PROMPT_TOPIC = "prompt-topic"  # Spring 서버로 설명을 전송할 토픽

# Kafka Producer 설정 (Flask -> Spring)
producer = KafkaProducer(
    bootstrap_servers=[KAFKA_BROKER],
    value_serializer=lambda v: json.dumps(v).encode("utf-8")
)


# Kafka Consumer 설정 (Spring -> Flask)
consumer = KafkaConsumer(
    IMAGE_TOPIC,
    bootstrap_servers=[KAFKA_BROKER],
    auto_offset_reset="earliest",
    enable_auto_commit=True,
    group_id="flask-consumer-group",
    value_deserializer=lambda x: json.loads(x.decode("utf-8"))
)


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

def consume_and_generate_prompt():
    for message in consumer:
        image_data = message.value
        image_url = image_data.get("image_url")

        if image_url:
            print(f"Image url: {image_url}")

            # generated_prompt = generate_prompt(image_url)
            generated_prompt = "test_prompt"
            print(f"Generated prompt: {generated_prompt}")

            prompt_data = {
                "image_id": image_data.get("image_id"),
                "prompt_text": generated_prompt,
            }

            producer.send(PROMPT_TOPIC, value = prompt_data)
            print("Send generated prompt")
if __name__ == "__main__":
    # Kafka Consumer를 별도의 스레드에서 실행
    from threading import Thread
    consumer_thread = Thread(target=consume_and_generate_prompt)
    consumer_thread.start()

    app.run(host="0.0.0.0", port=5001)
