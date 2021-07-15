# from transformers import AutoTokenizer, RobertaModel

# model = RobertaModel.from_pretrained('sinhala-roberta-mc4', from_flax=True)
# tokenizer = AutoTokenizer.from_pretrained('sinhala-roberta-mc4')

# tokenizer.save_pretrained('sinhala-roberta-mc4')
# model.save_pretrained('sinhala-roberta-mc4')

from transformers import GPT2Model, FlaxGPT2Model, AutoTokenizer
import torch
import numpy as np
import jax
import jax.numpy as jnp
jax.config.update('jax_platform_name', 'cpu')
MODEL_PATH = "Sinhala-gpt2"
model = FlaxGPT2Model.from_pretrained(MODEL_PATH)
def to_f32(t):
        return jax.tree_map(lambda x: x.astype(jnp.float32) if x.dtype == jnp.bfloat16 else x, t)
model.params = to_f32(model.params)
model.save_pretrained(MODEL_PATH)
pt_model = GPT2Model.from_pretrained(MODEL_PATH, from_flax=True).to('cpu')
# input_ids = np.asarray(2 * [128 * [0]], dtype=np.int32)
# input_ids_pt = torch.tensor(input_ids)
# logits_pt = pt_model(input_ids_pt).logits
# print(logits_pt)
# logits_fx = model(input_ids).logits
# print(logits_fx)
pt_model.save_pretrained(MODEL_PATH)
# also save tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
tokenizer.save_pretrained(MODEL_PATH)
