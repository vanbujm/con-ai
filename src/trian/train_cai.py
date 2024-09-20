import os
from dotenv import load_dotenv
from datasets import load_dataset
from unsloth import FastMistralModel
from trl import SFTTrainer
from transformers import TrainingArguments
from huggingface_hub import login
import json
import random
from datasets import Dataset
from datasets import concatenate_datasets
import torch

load_dotenv()

HUGGING_FACE_ACCESS_TOKEN = os.getenv('HUGGING_FACE_ACCESS_TOKEN')
login()

max_seq_length = 4096  # Can change to whatever number <= 4096
dtype = None  # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True  # Use 4bit quantization to reduce memory usage. Can be False.

model, tokenizer = FastMistralModel.from_pretrained(
    model_name="mistralai/Mistral-7B-Instruct-v0.1",  # You can change this to any Llama model!
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
    # trust_remote_code=True,
    token=HUGGING_FACE_ACCESS_TOKEN,
)
model = FastMistralModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj", ],
    lora_alpha=16,
    lora_dropout=0,  # Currently only supports dropout = 0
    bias="none",  # Currently only supports bias = "none"
    use_gradient_checkpointing=True,
    random_state=3407,
    max_seq_length=max_seq_length,
)

dataset = load_dataset("HuggingFaceH4/ultrachat_200k", token=HUGGING_FACE_ACCESS_TOKEN)

ultrachat_total_rows = len(dataset["train_sft"])

cai_data = []
with open('data/processed.jsonl') as f:
    for line in f:
        cai_obj = json.loads(line)
        cai_data.append({
            "prompt": cai_obj["initialPrompt"],
            "messages": [
                {"content": cai_obj["initialPrompt"], "role": "user"},
                {"content": cai_obj["revisionResponse"], "role": "assistant"}
            ]
        })

cai_data = Dataset.from_list(cai_data)

cai_data = cai_data.shuffle(seed=42)

train_sft = dataset["train_sft"]
train_sft = train_sft.shuffle(seed=42)
test_sft = dataset["test_sft"]
test_sft = test_sft.shuffle(seed=42)

# i choose 50k sample for training and 2k for test
train_sft_subset = train_sft.select(range(150000))
test_sft_subset = test_sft.select(range(3000))

# Get the same proportion of the CAI data as the Ultrachat data
cai_train_sft = cai_data.select(range(int(len(train_sft_subset)/ultrachat_total_rows * len(cai_data))))
cai_test_sft = cai_data.select(range(int(len(test_sft_subset)/ultrachat_total_rows * len(cai_data))))

print("Ultrachat Total Rows: ", ultrachat_total_rows)
print("Ultrachat Train Length: ", len(train_sft_subset))
print("Ultrachat Test Length: ", len(test_sft_subset))
print("CAI Data Length: ", len(cai_data))
print("CAI Train Length: ", len(cai_train_sft))
print("CAI Test Length: ", len(cai_test_sft))

train_sft_subset_len = len(train_sft_subset) - len(cai_train_sft)
train_sft_subset = concatenate_datasets([train_sft_subset.select(range(train_sft_subset_len)), cai_train_sft])

test_sft_subset_len = len(test_sft_subset) - len(cai_test_sft)
test_sft_subset = concatenate_datasets([test_sft_subset.select(range(test_sft_subset_len)), cai_test_sft])

print("Combined Train Length: ", len(train_sft_subset))
print("Combined Test Length: ", len(test_sft_subset))

def formatting_func(example):
    formatted_messages = []

    for message in example['messages']:
        content = message['content']
        role = message['role']
        formatted_message = {"role": role, "content": content}
        formatted_messages.append(formatted_message)

    return {"text": "\n".join([str(msg) for msg in formatted_messages])}

train_sft = train_sft_subset.map(formatting_func)
test_sft = test_sft_subset.map(formatting_func)


HAS_BFLOAT16 = torch.cuda.is_bf16_supported()
learning_rate = 1e-4
weight_decay = 0.01
warmup_steps = 10
lr_scheduler_type = "linear"
optimizer = "adamw_8bit"
random_state = 3407

argument = TrainingArguments(
    per_device_train_batch_size = 1,
    gradient_accumulation_steps = 4,
    warmup_steps = warmup_steps,
    max_steps = 240,
    learning_rate = learning_rate,
    fp16 = not HAS_BFLOAT16,
    bf16 = HAS_BFLOAT16,
    logging_steps = 1,
    output_dir = "outputs",
    optim = optimizer,
    weight_decay = weight_decay,
    lr_scheduler_type = lr_scheduler_type,
    seed = random_state,
    report_to = "none",
)

trainer = SFTTrainer(
    model = model,
    train_dataset=train_sft,
    eval_dataset=test_sft,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    args = argument,
)

trainer.train()

trainer.save_model("./ultrachat_cai")

trainer.push_to_hub("vanbujm/ultrachat_cai")
tokenizer.push_to_hub("vanbujm/ultrachat_cai")
