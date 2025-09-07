import torch.nn as nn
from transformers import BertModel, BertTokenizer
import torchvision.models as vision_models
import torch
from meld_dataset import MELDDataset


class TextEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        # Path to your offline BERT model folder
        bert_path = r"C:\Users\v-abhishek.tg\OneDrive - Aurigo Software Technologies Inc\Desktop\MP\models\bert_model_uncased"

        #  Load from local folder only
        self.bert = BertModel.from_pretrained(bert_path, local_files_only=True)
        self.tokenizer = BertTokenizer.from_pretrained(bert_path, local_files_only=True)

        for param in self.bert.parameters():
            param.requires_grad = False

        self.projection = nn.Linear(768, 256)

    def forward(self, input_ids, attention_mask):
        # Extract BERT embeddings
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
        # Input shape: (batch_size, frames, channels, height, width)
        x = x.transpose(1, 2)  # -> [batch_size, channels, frames, height, width]
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
        # Input shape: (batch_size, channels, time_steps)
        x = x.squeeze(1)  # Remove channel dimension if it's 1
        features = self.conv_layers(x)  # Shape: (batch_size, 128, 1)
        return self.projection(features.squeeze(-1))  # Shape: (batch_size, 256)


class MultimodalSentimentalModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.text_encoder = TextEncoder()
        self.video_encoder = VideoEncoder()
        self.audio_encoder = AudioEncoder()

        # Fusion layer
        self.fusion_layer = nn.Sequential(
            nn.Linear(256 * 3, 256),
            # nn.BatchNorm1d(256), 
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.3),
        )

        # Classification layers
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

        # Concatenate features
        combined_features = torch.cat(
            [text_features, video_features, audio_features], dim=1
        )  # [batch_size, 128*3]

        fused_features = self.fusion_layer(combined_features)  # [batch_size, 256]
        emotion_logits = self.emo_classifier(fused_features)
        sentiment_logits = self.sent_classifier(fused_features)

        return {
            'emotions': emotion_logits,
            'sentiments': sentiment_logits
        }


if __name__ == "__main__":
    dataset = MELDDataset(
        r'C:\Users\v-abhishek.tg\OneDrive - Aurigo Software Technologies Inc\Desktop\MP\dataset\train\train_sent_emo.csv',
        r'C:\Users\v-abhishek.tg\OneDrive - Aurigo Software Technologies Inc\Desktop\MP\dataset\train\train_splits'
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

        for i, prob in enumerate(emotion_logits):
            print(f"{emotion_map[i]}: {prob:.4f}")
        for i, prob in enumerate(sentiment_logits):
            print(f"{sentiment_map[i]}: {prob:.4f}")

    print("Done")
