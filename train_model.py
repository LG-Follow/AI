# train_model.py
from utils.blip_training import train_blip_model

# 모델 학습
num_epochs = 1
trained_model, processor = train_blip_model(num_epochs=num_epochs)

# 모델 저장
trained_model.save_pretrained("project_directory/models/trained_model")
processor.save_pretrained("project_directory/models/trained_model")
