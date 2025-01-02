import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
from torchvision.models import vit_b_16
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import optuna

class TextProcessor:
    def __init__(self, model_name='bert-base-uncased'):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.bert_model = BertModel.from_pretrained(model_name).eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.bert_model.to(self.device)

    def process_text(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512).to(self.device)
        with torch.no_grad():
            embeddings = self.bert_model(**inputs).last_hidden_state.mean(dim=1)
        return embeddings.cpu().numpy()

class ImageProcessor:
    def __init__(self):
        self.model = vit_b_16(pretrained=True)
        self.model.heads = nn.Identity()
        self.model.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.transform = Compose([
            Resize((224, 224)),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def process_image(self, image_path):
        from PIL import Image
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            features = self.model(image).cpu().numpy()
        return features

class BloodProcessor:
    def process_blood_data(self, blood_data):
        normalized_data = (blood_data - np.mean(blood_data, axis=0)) / np.std(blood_data, axis=0)
        return normalized_data

class UnifiedHealthModel(nn.Module):
    def __init__(self):
        super(UnifiedHealthModel, self).__init__()
        self.text_net = nn.Linear(768, 256)
        self.image_net = nn.Linear(768, 256)
        self.blood_net = nn.Linear(10, 256)
        self.final_net = nn.Sequential(
            nn.Linear(256 * 3, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, text_features, image_features, blood_features):
        text_out = self.text_net(text_features)
        image_out = self.image_net(image_features)
        blood_out = self.blood_net(blood_features)
        combined = torch.cat([text_out, image_out, blood_out], dim=1)
        output = self.final_net(combined)
        return output

class UnifiedHealthTrainer:
    def __init__(self):
        self.model = UnifiedHealthModel().to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def train(self, text_data, image_data, blood_data, labels, epochs=10, lr=1e-4):
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.BCELoss()

        for epoch in range(epochs):
            optimizer.zero_grad()
            text_data = torch.tensor(text_data, dtype=torch.float32).to(self.device)
            image_data = torch.tensor(image_data, dtype=torch.float32).to(self.device)
            blood_data = torch.tensor(blood_data, dtype=torch.float32).to(self.device)
            labels = torch.tensor(labels, dtype=torch.float32).unsqueeze(1).to(self.device)

            outputs = self.model(text_data, image_data, blood_data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}")

    def evaluate(self, text_data, image_data, blood_data, labels):
        self.model.eval()
        with torch.no_grad():
            text_data = torch.tensor(text_data, dtype=torch.float32).to(self.device)
            image_data = torch.tensor(image_data, dtype=torch.float32).to(self.device)
            blood_data = torch.tensor(blood_data, dtype=torch.float32).to(self.device)
            labels = torch.tensor(labels, dtype=torch.float32).unsqueeze(1).to(self.device)

            outputs = self.model(text_data, image_data, blood_data)
            predictions = (outputs > 0.5).float()
            print(classification_report(labels.cpu().numpy(), predictions.cpu().numpy()))

if __name__ == "__main__":
    text_processor = TextProcessor()
    image_processor = ImageProcessor()
    blood_processor = BloodProcessor()

    blood_data = np.random.rand(100, 10)
    blood_labels = np.random.randint(0, 2, 100)
    text_descriptions = ["Patient has mild chest pain"] * 100
    image_paths = ["example_xray.jpg"] * 100

    text_features = np.vstack([text_processor.process_text(desc) for desc in text_descriptions])
    image_features = np.vstack([image_processor.process_image(path) for path in image_paths])
    blood_features = blood_processor.process_blood_data(blood_data)

    X_text_train, X_text_test, X_img_train, X_img_test, X_blood_train, X_blood_test, y_train, y_test = train_test_split(
        text_features, image_features, blood_features, blood_labels, test_size=0.2, random_state=42)

    trainer = UnifiedHealthTrainer()
    trainer.train(X_text_train, X_img_train, X_blood_train, y_train)
    trainer.evaluate(X_text_test, X_img_test, X_blood_test, y_test)
