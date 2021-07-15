model_dir="./wav2vec2-si-base"

from transformers import Wav2Vec2FeatureExtractor
config = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-base", return_attention_mask=True)
config.save_pretrained(model_dir)
