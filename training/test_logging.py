from collections import namedtuple
import torch 
from torch.utils.data import DataLoader, Dataset
from models import  MultimodalSentimentalModel , MultimodalTrainer


def test_logging():
    Batch = namedtuple('Batch' , ['text_inputs' , 'video_frames' , 'audio_features'])
    mock_batch = Batch(
        text_inputs={'input_ids':torch.ones(1),'attention_mask':torch.ones(1)},
        video_frames = torch.ones(1),
        audio_features=torch.ones(1)
    )
    mock_loader = DataLoader([mock_batch])

    model = MultimodalSentimentalModel()
    trainer = MultimodalTrainer(model, mock_loader, mock_loader)
    train_losses = {
        'total':2.5,
        'emotion':1.0,
        'sentiment':1.5
    }
    trainer.log_metrics(train_losses,phase="train" , metrics=None)


    val_losses = {
        'total':1.5,
        'emotion':0.5,
        'sentiment':1.0   
    }

    val_metrics={
        'emotion_precision':0.65,
        'emotion_accuracy':0.75,
        'sentiment_precision':0.85,
        'sentiment_accuracy':0.95
    }
    trainer.log_metrics(val_losses, phase="val", metrics=val_metrics)








if __name__ == "__main__":
    test_logging() 