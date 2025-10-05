# %%
## 模型初始化
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from datasets import load_dataset
import torch
import json
import os
dataset_path = "./ccfbdci.jsonl"
dataset_new_path = "./ccfbdci_ner.jsonl"
model_path = "./model" # 使用qwen2.5-1.5b-instruct
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", dtype=torch.bfloat16)
model.enable_input_require_grads()

# %%
## 数据集构造函数 & 预测生成函数
def dataset_transfer(origin_path, new_path):
    messages = []
    match_name = ["地名", "人名", "地理实体", "组织"]
    with open(origin_path, "r") as f:
        for line in f:
            data = json.loads(line)
            input_text = data["text"]
            entities = data["entities"]
            entity_sentence = ""
            for entity in entities:
                entity_json = dict(entity)
                entity_text = entity_json["entity_text"]
                entity_names = entity_json["entity_names"]
                for name in entity_names:
                    if name in match_name:
                        entity_label = name
                        break
                entity_sentence += f"""{{"entity_text": "{entity_text}", "entity_label": "{entity_label}"}}"""
            if entity_sentence == "":
                entity_sentence = "没有找到任何实体"
            
            message = {
                "instruction": """你是一个文本实体识别领域的专家，你需要从给定的句子中提取 地点; 人名; 地理实体; 组织 实体. 以 json 格式输出, 如 {"entity_text": "南京", "entity_label": "地理实体"} 注意: 1. 输出的每一行都必须是正确的 json 字符串. 2. 找不到任何实体时, 输出"没有找到任何实体". """,
                "input": f"文本:{input_text}",
                "output": entity_sentence
            }
            messages.append(message)
    with open(new_path, "w", encoding="utf-8") as f:
        for message in messages:
            f.write(json.dumps(message, ensure_ascii=False) + "\n")

def process_func(example):
    MAX_LENGTH = 384
    input_ids, attention_mask, labels = [], [], []

    instruction = tokenizer.apply_chat_template(
        conversation=[
            {"role": "user", "content": example["instruction"]},
            {"role": "user", "content": example["input"]},
        ],
        add_generation_prompt=True,
        tokenize=False,
    )
    instruction = tokenizer(instruction, add_special_tokens=False)
    response = tokenizer(example["output"] + tokenizer.eos_token, add_special_tokens=False)
    input_ids = instruction["input_ids"] + response["input_ids"]
    attention_mask = instruction["attention_mask"] + response["attention_mask"]
    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"]
    if len(input_ids) > MAX_LENGTH:
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }

def predict(message, model, tokenizer):
    inputs = tokenizer.apply_chat_template(
        conversation=[
            {"role": "user", "content": message["instruction"]},
            {"role": "user", "content": message["input"]},
        ],
        add_generation_prompt=True,
        tokenize=False,
    )
    inputs = tokenizer(inputs, return_tensors="pt").to(model.device)
    generation_output = model.generate(**inputs, max_new_tokens=512)
    outputs = generation_output[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(outputs, skip_special_tokens=True)
    return response


# %%
## 构造数据集
if not os.path.exists(dataset_new_path):
    dataset_transfer(dataset_path, dataset_new_path)
train_dataset = load_dataset("json", data_files=dataset_new_path)
train_dataset = train_dataset.map(process_func, remove_columns=train_dataset["train"].column_names)

# %%
## 构造lora模型
from peft import LoraConfig, get_peft_model, TaskType
config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj",  "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    r=8,
    lora_alpha=32,
    lora_dropout=0.1
)
model = get_peft_model(model, config)

# %%
## 训练
from transformers.data.data_collator import DataCollatorForSeq2Seq
args = TrainingArguments(
    output_dir="./output/Qwen2-NER",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=4,
    logging_steps=100,
    num_train_epochs=2,
    save_steps=100,
    learning_rate=1e-4,
    save_on_each_node=True,
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset["train"],
    data_collator=DataCollatorForSeq2Seq(tokenizer),
)

trainer.train()

# %%q
## 抽样展示结果
import pandas as pd
model.eval()
df = pd.read_json(dataset_new_path, lines=True)
test_df = df[:int(len(df) * 0.1)].sample(20)
for index, row in test_df.iterrows():
    response = predict(row, model, tokenizer)
    print(response)
