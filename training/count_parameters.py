from models import MultimodalSentimentalModel , MultimodalTrainer

def count_parameters(model):
    params_dict = {
        'text_encoder':0,
        'video_encoder':0,
        'audio_encoder':0,
        'fusion_layer':0,
        'emotion_classifier':0,
        'sentiment_classifier':0,
    }

    total_params = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            total_params += param.numel()
            params_count = param.numel()
            total_params+= param.numel()

            if 'text_encoder' in name:
                params_dict['text_encoder'] += params_count
            elif 'video_encoder' in name:
                params_dict['video_encoder'] += params_count
            elif 'audio_encoder' in name:
                params_dict['audio_encoder'] += params_count
            elif 'fusion_layer' in name:
                params_dict['fusion_layer'] += params_count
            elif 'emo_classifier' in name:
                params_dict['emotion_classifier'] += params_count
            elif 'sent_classifier' in name:
                params_dict['sentiment_classifier'] += params_count

    return params_dict , total_params

if __name__ == "__main__":
    model = MultimodalSentimentalModel()
    params_dict, total_params = count_parameters(model)

    print("Parameter Count by Component:")
    for component, count in params_dict.items():
        print(f"{component:20s}: {count:,} parameters")

    print(f"\nTotal Trainable Parameters: {total_params:,} parameters")