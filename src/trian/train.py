from datasets import load_dataset
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
from trl import SFTConfig, SFTTrainer

dataset = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft")

sft_config = SFTConfig(
    dataset_text_field="text",
    max_seq_length=512,
    output_dir="/tmp",
)

model = AutoModelForSequenceClassification.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1", num_labels=5)

trainer = SFTTrainer(
    model,
    train_dataset=dataset,
    args=sft_config,
)
trainer.train()