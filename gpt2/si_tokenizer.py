from datasets import load_dataset, DatasetDict
from tokenizers import trainers, Tokenizer, normalizers, ByteLevelBPETokenizer

model_dir = "Sinhala-gpt2"  # ${MODEL_DIR}

# load dataset
# dataset = load_dataset("mc4", "si", split="train")
dataset = DatasetDict.load_from_disk('../data/')["train"]
# Instantiate tokenizer
tokenizer = ByteLevelBPETokenizer()

def batch_iterator(batch_size=1000):
    for i in range(0, len(dataset), batch_size):
        yield dataset[i: i + batch_size]["text"]

# Customized training
tokenizer.train_from_iterator(batch_iterator(), vocab_size=50265, min_frequency=2, special_tokens=[
    "<s>",
    "<pad>",
    "</s>",
    "<unk>",
    "<mask>",
])

# Save files to disk
tokenizer.save(f"{model_dir}/tokenizer.json")
