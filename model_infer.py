import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

torch.set_default_device("cuda")

model = AutoModelForCausalLM.from_pretrained("xiangxinai/Xiangxin-3B", 
                          torch_dtype="auto", 
                          trust_remote_code=True)

tokenizer = AutoTokenizer.from_pretrained("xiangxinai/Xiangxin-3B", 
                      trust_remote_code=True)

inputs = tokenizer("年轻人不应该再努力了，年轻人就应该享受生活，", 
        return_tensors="pt", return_attention_mask=False)

outputs = model.generate(**inputs, max_length=100, 
                         do_sample=True, 
                         temperature=1.0, 
                         pad_token_id=tokenizer.eos_token_id)

text = tokenizer.batch_decode(outputs)[0]

print(text)