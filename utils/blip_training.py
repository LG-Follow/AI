# utils/blip_training.py
import torch
from transformers import BlipForConditionalGeneration, BlipProcessor
from torch.utils.data import DataLoader
from utils.data_processing import load_flickr30k_data, preprocess_data
from utils.collate import collate_fn

# utils/blip_training.py

def train_blip_model(num_epochs=1, batch_size=16, learning_rate=1e-5):
    # 모델 및 프로세서 초기화
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    model.to("cuda")

    # 데이터셋 로드 및 전처리
    dataset = load_flickr30k_data()
    processed_dataset = dataset.map(lambda x: preprocess_data(x, processor))

    # DataLoader 설정
    dataloader = DataLoader(processed_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # 학습 루프
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for step, (input_ids, attention_mask, pixel_values, captions) in enumerate(dataloader):
            input_ids = input_ids.to("cuda")
            attention_mask = attention_mask.to("cuda")
            pixel_values = pixel_values.to("cuda")

            # attention_mask 확인용 출력 (학습 시에는 삭제)
            print("input_ids shape:", input_ids.shape)
            print("attention_mask shape:", attention_mask.shape)
            print("pixel_values shape:", pixel_values.shape)

            # 모델 출력 및 손실 계산
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values, labels=input_ids)
            loss = outputs.loss
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()

            # 배치 손실 출력
            print(f"Epoch [{epoch+1}/{num_epochs}], Step [{step+1}/{len(dataloader)}], Batch Loss: {loss.item()}")

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Average Loss: {avg_loss:.4f}")

    return model, processor


