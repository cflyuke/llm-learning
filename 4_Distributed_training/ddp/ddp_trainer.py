import os
import evaluate
from datasets import load_dataset
import torch.distributed as dist 
from transformers import DataCollatorWithPadding
from transformers import Trainer, TrainingArguments, BertTokenizer, BertForSequenceClassification


def prepare_dataset(tokenizer):
    dataset = load_dataset("csv", data_files="./ChnSentiCorp_htl_all.csv", split="train")
    dataset = dataset.filter(lambda x: x["review"] is not None)
    datasets = dataset.train_test_split(test_size=0.1, seed=42)
    def process_function(examples):
        tokenized_examples = tokenizer(examples["review"], max_length=128, truncation=True)
        tokenized_examples["labels"] = examples["label"]
        return tokenized_examples
    tokenized_datasets = datasets.map(process_function, batched=True, remove_columns=datasets["train"].column_names)
    return tokenized_datasets

def compute_metrics(eval_predict):
    acc_metric = evaluate.load("./accuracy.py")
    f1_metric = evaluate.load("./f1.py")
    predictions, labels = eval_predict
    predictions = predictions.argmax(axis=-1)
    acc = acc_metric.compute(predictions=predictions, references=labels)
    f1 = f1_metric.compute(predictions=predictions, references=labels)
    acc.update(f1)
    return acc

def main():
    checkpoint = "./model/hfl/rbt3"
    tokenizer = BertTokenizer.from_pretrained(checkpoint)
    tokenized_datasets = prepare_dataset(tokenizer)
    model = BertForSequenceClassification.from_pretrained(checkpoint, num_labels=2)

    dist.init_process_group(backend='nccl', device_id=int(os.environ["LOCAL_RANK"]))

    train_args = TrainingArguments(
        output_dir="./checkpoints",      # 输出文件夹
        per_device_train_batch_size=32,  # 训练时的batch_size
        per_device_eval_batch_size=128,  # 验证时的batch_size
        logging_steps=100,                # log 打印的频率
        eval_strategy="epoch",           # 评估策略
        save_strategy="epoch",           # 保存策略
        save_total_limit=3,              # 最大保存数
        learning_rate=2e-5,              # 学习率
        weight_decay=0.01,               # weight_decay
        metric_for_best_model="f1",      # 设定评估指标
        load_best_model_at_end=True,      # 训练完成后加载最优模型
    )

    trainer = Trainer(
        model=model, 
        args=train_args, 
        train_dataset=tokenized_datasets["train"], 
        eval_dataset=tokenized_datasets["test"], 
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics
    )

    trainer.train()
    dist.destroy_process_group()

if __name__ == "__main__":
    main()