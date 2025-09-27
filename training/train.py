import os
import argparse
import torchaudio
import torch
from meld_dataset import MELDDataset, prepare_dataloader
from models import MultimodalSentimentalModel, MultimodalTrainer
from meld_dataset import prepare_dataloader
import random
import json
import tqdm
from install_ffmpeg import install_ffmpeg
import sys

# AWS SageMaker
SM_MODEL_DIR = os.environ.get('SM_MODEL_DIR', '.')
SM_CHANNEL_TRAINING = os.environ.get('SM_CHANNEL_TRAINING', "/opt/ml/input/data/training")
SM_CHANNEL_TESTING = os.environ.get('SM_CHANNEL_TESTING', "/opt/ml/input/data/testing")


os.environ['PYTORCH_CUDA_ALLOC_CONF'] = "expandable_segments: True"


def parse_args():
    parser = argparse.ArgumentParser(description="Multimodal Sentiment Analysis Training")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training and validation")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate for the optimizer")

    #data directories
    parser.add_argument("--train_data_dir", type=str, default=SM_CHANNEL_TRAINING, help="Directory for training data")
    parser.add_argument("--val-dir", type=str, default=SM_CHANNEL_TESTING, help="Directory for validation data")
    parser.add_argument("--test-dir", type=str, default=SM_CHANNEL_TESTING, help="Directory for testing data")
    parser.add_argument("--model_dir", type=str, default=SM_MODEL_DIR, help="Directory to save the trained model") 

    return parser.parse_args()

def main():
    #INSTALL FFMPEG  
    if not install_ffmpeg():
        print(" EEEOR: FFmpeg instalation failed , cannot continue training....")
        sys.exit(1)

 
    print("Available Audio Backends:")
    print(str(torchaudio.list_audio_backends()))

    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #track initial GPU Memory if Available
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        initial_memory = torch.cuda.max_memory_allocated() / 1024**3
        print(f"Initial GPU Memory USED: {initial_memory / (1024 ** 3):.4f} GB")

    train_loader, val_loader, test_loader = prepare_dataloader(
        train_csv=os.path.join(args.train_dir, "train_sent_emo.csv"),
        train_video_dir=os.path.join(args.train_dir, "train_splits"),
        dev_csv=os.path.join(args.val_dir, "dev_sent_emo.csv"),
        dev_video_dir=os.path.join(args.val_dir, "dev_splits_complete"),
        test_csv=os.path.join(args.test_dir, "test_sent_emo.csv"),
        test_video_dir=os.path.join(args.test_dir, "output_repeated_splits_test"),
        batch_size=args.batch_size
    )

    print(f""" Training CSV Path : {os.path.join(args.train_dir, "train_sent_emo.csv")}""")
    print(f"""Training Video Directory : {os.path.join(args.train_dir, "train_splits")}""")

    model = MultimodalSentimentalModel().to(device)
    trainer = MultimodalTrainer(model, train_loader, val_loader, device=device)
    best_val_loss = float('inf')

    metrics_data={
        'train_losses':[],
        'val_losses':[],
        'epochs':[]
    }


    for epoch in tqdm(args.epochs):
        train_loss = trainer.train_epoch()
        val_loss , val_metrics =trainer.evaluate(val_loader)    
        

        metrics_data['train_losses'].append(train_loss["total"])
        metrics_data['val_losses'].append(val_loss["total"])
        metrics_data['epochs'].append(epoch)


        #Log metrics  in SageMaker formate
        print(json.dumps({
            "metrics":[
                {"Name":"train_loss", "Value":train_loss['total']},
                {"Name":"validation:loss", "Value":val_loss['total']},
                {"Name":"validation:emotion_precision", "Value":val_metrics['emotion_precision']},
                {"Name":"validation:emotion_accuracy", "Value":val_metrics['emotion_accuracy']},
                {"Name":"validation:sentiment_precision", "Value":val_metrics['sentiment_precision']},
                {"Name":"validation:sentiment_accuracy", "Value":val_metrics['sentiment_accuracy']} 
                

            ]
        }))

        if torch.cuda .is_available():
            memory_used = torch.cuda.max_memory_allocated() / 1024**3
            print("peak GPU Memory USED: {:.4f} GB".format(memory_used))

            if val_loss["total"] < best_val_loss:
                best_val_loss = val_loss["total"]
                model_path = os.path.join(args.model_dir, "model.pth")
                torch.save(model.state_dict(), model_path)
               

    # after Training is complete eveluate on test set

    print("Evaluate on test set")
    test_loss , test_metrics = trainer.evaluate(test_loader , phase="test")
    metrics_data["test_loss"] = test_loss["total"]


    print(json.dumps({
        "metrics":[
            {"Name": "test:loss" , "Value":test_loss["total"]},
            {"Name":"test:emotion_accuracy", "Value": test_metrics["emotion_accuracy"]} ,
            {"Name":"test:sentiment_accuracy", "Value": test_metrics["sentiment_accuracy"]} ,
            {"Name":"test:emotion_precision", "Value": test_metrics["emotion_precison"]} ,
            {"Name":"test:sentiment_precision", "Value": test_metrics["sentiment_precision"]} ,
        
        ]
    }))






if __name__ == "__main__":
    main()
  