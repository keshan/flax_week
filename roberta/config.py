from transformers import RobertaConfig

model_dir = "./Sinhala-roberta"  # ${MODEL_DIR}

config = RobertaConfig.from_pretrained("roberta-base")
config.save_pretrained(model_dir)
