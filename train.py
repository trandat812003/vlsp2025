from datasets import load_dataset
from unsloth import FastLanguageModel
from trl import SFTConfig, SFTTrainer
from transformers import TrainingArguments

# Load dataset
data_train = load_dataset("json", data_files={"train": "data/train.json"}, split="train")
data_test = load_dataset("json", data_files={"test": "data/test.json"}, split="test")

# # Format dataset
# def format_example(example):
#     prompt = example['instruction']
#     if example.get('input'):
#         prompt += "\n" + example['input']
#     return {"prompt": prompt, "completion": example['output']}

def format_example(example):
    prompt = example['instruction']
    if example.get('input'):
        prompt += "\n" + example['input']
    prompt += "\n" + example['output']  # append output vào prompt
    return {"text": prompt}

dataset_train = data_train.map(lambda ex: format_example(ex), remove_columns=data_train.column_names)
dataset_test = data_test.map(lambda ex: format_example(ex), remove_columns=data_test.column_names)

# Load model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen2.5-3B",
    max_seq_length=2048,
    # load_in_4bit=True,
    # full_finetuning=False,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0.1,
    bias="none",
    use_gradient_checkpointing="unsloth",  # giảm bộ nhớ hơn bằng cơ chế checkpoint
    random_state=42,
    max_seq_length=2048,
)


# Cấu hình huấn luyện
# config = SFTConfig(
#     max_seq_length=2048,
#     per_device_train_batch_size=2,
#     gradient_accumulation_steps=4,
#     num_train_epochs=3,                # train 3 epoch
#     evaluation_strategy="epoch",       # test sau mỗi epoch
#     save_strategy="epoch",             # lưu checkpoint sau mỗi epoch
#     logging_steps=10,
#     output_dir="output",
#     optim="adamw_8bit",
#     learning_rate=2e-4,
#     seed=42,
#     report_to=[],
# )

training_args = TrainingArguments(
    output_dir="output",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    learning_rate=2e-4,
    logging_steps=100,
    optim="adamw_torch_fused",
    seed=42,
    report_to=[],
)

# Trainer
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset_train,
    eval_dataset=dataset_test,           # thêm eval_dataset
    args=training_args,
    # formatting_func=lambda ex: (ex["prompt"], ex["completion"]),
)

trainer.train()
