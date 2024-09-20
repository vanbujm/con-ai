from transformers import AutoModel, AutoTokenizer
from huggingface_hub import login


max_seq_length = 4096  # Can change to whatever number <= 4096
dtype = None  # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True  # Use 4bit quantization to reduce memory usage. Can be False.

model = AutoModel.from_pretrained('./ultrachat_cai', load_in_4bit=True, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained('./ultrachat_cai')


login()
model.push_to_hub("vanbujm/ultrachat_cai")
tokenizer.push_to_hub("vanbujm/ultrachat_cai")
