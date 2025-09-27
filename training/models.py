import torch.nn as nn
from transformers import BertModel, BertTokenizer
import torchvision.models as vision_models
import torch
from meld_dataset import MELDDataset
from sklearn.metrics import (
    classification_report, confusion_matrix, precision_score,
    recall_score, f1_score, accuracy_score
)
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import os


class TextEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        # Path to your offline BERT model folder (UPDATED)
        bert_path = r"C:\MP\models\bert_model_uncased"

        # Load from local folder only
        self.bert = BertModel.from_pretrained(bert_path, local_files_only=True)
        self.tokenizer = BertTokenizer.from_pretrained(bert_path, local_files_only=True)

        for param in self.bert.parameters():
            param.requires_grad = False

        self.projection = nn.Linear(768, 256)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output  # [CLS] token representation
        return self.projection(pooled_output)


class VideoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = vision_models.video.r3d_18(pretrained=True)
        for param in self.backbone.parameters():
            param.requires_grad = False
        num_ftrs = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(num_ftrs, 256),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

    def forward(self, x):
        x = x.transpose(1, 2)  # [batch, channels, frames, height, width]
        return self.backbone(x)


class AudioEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(64, 128, kernel_size=3),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        for param in self.conv_layers.parameters():
            param.requires_grad = False
        self.projection = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

    def forward(self, x):
        x = x.squeeze(1)  # remove channel if it's 1
        features = self.conv_layers(x)  # (batch, 128, 1)
        return self.projection(features.squeeze(-1))  # (batch, 256)


class MultimodalSentimentalModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.text_encoder = TextEncoder()
        self.video_encoder = VideoEncoder()
        self.audio_encoder = AudioEncoder()

        self.fusion_layer = nn.Sequential(
            nn.Linear(256 * 3, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.3),
        )

        self.emo_classifier = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 7)  # 7 emotion classes
        )

        self.sent_classifier = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 3)  # 3 sentiment classes
        )

    def forward(self, text_inputs, video_frames, audio_features):
        text_features = self.text_encoder(
            text_inputs['input_ids'], text_inputs['attention_mask']
        )
        video_features = self.video_encoder(video_frames)
        audio_features = self.audio_encoder(audio_features)

        combined_features = torch.cat(
            [text_features, video_features, audio_features], dim=1
        )
        fused_features = self.fusion_layer(combined_features)
        return {
            'emotions': self.emo_classifier(fused_features),
            'sentiments': self.sent_classifier(fused_features)
        }


class MultimodalTrainer:
    def __init__(self, model, train_loader, val_loader):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader

        print(f"Training dataset size: {len(train_loader.dataset):,}")
        print(f"Validation dataset size: {len(val_loader.dataset):,}")
        print(f"Batches per epoch: {len(train_loader):,}")

        timestamp = datetime.now().strftime('%b%d_%H-%M-%S')
        base_dir = '/opt/ml/output/tensorboard' if 'SM_MODEL_DIR' in os.environ else 'runs'
        log_dir = f"{base_dir}/run_{timestamp}"
        self.writer = SummaryWriter(log_dir=log_dir)
        self.global_step = 0

        self.optimizer = torch.optim.Adam([
            {'params': model.text_encoder.parameters(), 'lr': 8e-6},
            {'params': model.video_encoder.parameters(), 'lr': 8e-5},
            {'params': model.audio_encoder.parameters(), 'lr': 8e-5},
            {'params': model.fusion_layer.parameters(), 'lr': 5e-4},
            {'params': model.emo_classifier.parameters(), 'lr': 5e-4},
            {'params': model.sent_classifier.parameters(), 'lr': 5e-4}
        ], weight_decay=1e-5)

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.1, patience=2
        )

        self.emotion_criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
        self.sentiment_criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
        self.current_train_losses = None

    def log_metrics(self, losses, metrics=None, phase="train"):
        if phase == "train":
            self.current_train_losses = losses
        else:
            self.writer.add_scalar('loss/total/train', self.current_train_losses['total'], self.global_step)
            self.writer.add_scalar('loss/total/val', losses['total'], self.global_step)
            self.writer.add_scalar('loss/emotion/train', self.current_train_losses['emotion'], self.global_step)
            self.writer.add_scalar('loss/emotion/val', losses['emotion'], self.global_step)
            self.writer.add_scalar('loss/sentiment/train', self.current_train_losses['sentiment'], self.global_step)
            self.writer.add_scalar('loss/sentiment/val', losses['sentiment'], self.global_step)

        if metrics:
            for k, v in metrics.items():
                self.writer.add_scalar(f"{phase}/{k}", v, self.global_step)

    # train_epoch and evaluate unchanged ...


if __name__ == "__main__":
    # UPDATED PATHS
    dataset = MELDDataset(
        r"C:\MP\dataset\train\train_sent_emo.csv",
        r"C:\MP\dataset\train\train_splits"
    )
    sample = dataset[0]
    model = MultimodalSentimentalModel()

    text_inputs = {
        'input_ids': sample['text_inputs']['input_ids'].unsqueeze(0),
        'attention_mask': sample['text_inputs']['attention_mask'].unsqueeze(0)
    }
    videoframes = sample['video_frames'].unsqueeze(0)
    audiofeatures = sample['audio_features'].unsqueeze(0)

    with torch.inference_mode():
        outputs = model(text_inputs, videoframes, audiofeatures)

        emotion_logits = torch.softmax(outputs['emotions'], dim=1)[0]
        sentiment_logits = torch.softmax(outputs['sentiments'], dim=1)[0]

        emotion_map = {
            0: 'sadness', 1: 'anger', 2: 'fear', 3: 'joy',
            4: 'neutral', 5: 'disgust', 6: 'surprise'
        }
        sentiment_map = {0: 'negative', 1: 'neutral', 2: 'positive'}

        print("\nEmotion probabilities:")
        for i, prob in enumerate(emotion_logits):
            print(f"{emotion_map[i]}: {prob:.4f}")

        print("\nSentiment probabilities:")
        for i, prob in enumerate(sentiment_logits):
            print(f"{sentiment_map[i]}: {prob:.4f}")

    print(" Done")
