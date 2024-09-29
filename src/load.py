from tokenizers.pre_tokenizers import Whitespace
import torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer, BitsAndBytesConfig
from huggingface_hub import login
from dotenv import load_dotenv

load_dotenv()
# login()

# model = AutoModelForCausalLM.from_pretrained(
#     "ultrachat_cai",
#     quantization_config=bnb_config,
# )
#
# print(model)
#
# model = model.merge_and_unload()
# model.save_pretrained("ultrachat_cai_full")


#
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_compute_dtype=getattr(torch, "float16"),
    bnb_4bit_use_double_quant=False,
)

device_map = {"": 0}
lora_dir = "ultrachat_sai"
base_model_name = "mistralai/Mistral-7B-Instruct-v0.1"
tokenizer = AutoTokenizer.from_pretrained(lora_dir, trust_remote_code=True, use_fast=True, add_prefix_space=True)
tokenizer.pre_tokenizer = Whitespace()
model = AutoPeftModelForCausalLM.from_pretrained(lora_dir, quantization_config=bnb_config, return_dict=True,
                                                 device_map=device_map,
                                                 trust_remote_code=True, )
print("Loaded model")

model = model.merge_and_unload()
#
print("Merged model")

output_dir = lora_dir + "_full_2"
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

print("Saved model")
#
# model.push_to_hub("vanbujm/ultrachat_cai_full2")
# print("Pushed model")
# tokenizer.push_to_hub("vanbujm/ultrachat_cai_full2")
# print("Pushed tokenizer")
