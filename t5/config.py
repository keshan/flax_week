from transformers import T5Config

model_dir = "./sinhala-t5-small"  # ${MODEL_DIR}

config = T5Config.from_pretrained("t5-small")
config.save_pretrained(model_dir)
