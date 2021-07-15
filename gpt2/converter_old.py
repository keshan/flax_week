from transformers import AutoTokenizer, GPT2Model, TFGPT2Model

name = 'Sinhala-gpt2'

model_pt = GPT2Model.from_pretrained(name, from_flax=True)
model_pt.save_pretrained(name)

model_tf = TFGPT2Model.from_pretrained(name, from_pt=True)
tokenizer = AutoTokenizer.from_pretrained(name)

tokenizer.save_pretrained(name)
model_tf.save_pretrained(name)
