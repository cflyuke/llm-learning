import math
import torch
import pandas as pd
from torch.optim import AdamW
import torch.distributed as dist
from accelerate import Accelerator
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import BertForSequenceClassification, BertTokenizer

class MyDataset(Dataset):

    def __init__(self) -> None:
        super().__init__()
        self.data = pd.read_csv("./ChnSentiCorp_htl_all.csv")
        self.data = self.data.dropna()

    def __getitem__(self, index):
        return self.data.iloc[index]["review"], self.data.iloc[index]["label"]
    
    def __len__(self):
        return len(self.data)


def prepare_dataloader(checkpoint):
    dataset = MyDataset()
    trainset, validset = random_split(dataset, lengths=[0.9, 0.1], generator=torch.Generator().manual_seed(42))
    tokenizer = BertTokenizer.from_pretrained(checkpoint)
    def collate_fn(batch):
        texts, labels = [], []
        for item in batch:
            texts.append(item[0])
            labels.append(item[1])
        inputs = tokenizer(texts, max_length=128, padding="max_length", truncation=True, return_tensors="pt")
        inputs["labels"] = torch.tensor(labels)
        return inputs

    trainloader = DataLoader(trainset, batch_size=32, collate_fn=collate_fn, shuffle=True)
    validloader = DataLoader(validset, batch_size=64, collate_fn=collate_fn, shuffle=False)
    return trainloader, validloader


def prepare_model_and_optimizer(checkpoint):
    model = BertForSequenceClassification.from_pretrained(checkpoint)
    optimizer = AdamW(model.parameters(), lr=2e-5)
    return model, optimizer


def evaluate(model, validloader, accelerator: Accelerator):
    model.eval()
    acc_num = 0
    with torch.no_grad():
        for batch in validloader:
            output = model(**batch)
            pred = torch.argmax(output.logits, dim=-1)
            pred, refs = accelerator.gather_for_metrics((pred, batch["labels"]))
            acc_num += (pred.long() == refs.long()).float().sum()
    return acc_num / len(validloader.dataset)

def train(model, optimizer, trainloader, validloader, accelerator: Accelerator, resume=None, epoch=3, log_step=10, save_step=50):
    global_step = 0
    resume_epoch = 0
    resume_step = 0
    if resume:
        accelerator.load_state(resume)
        steps_per_epoch = math.ceil(len(trainloader) / accelerator.gradient_accumulation_steps)
        resume_step = global_step = int(resume.split("_")[-1])
        resume_epoch = resume_step // steps_per_epoch
        resume_step -= resume_epoch * steps_per_epoch

    for ep in range(resume_epoch, epoch):
        model.train()
        activeloader = trainloader
        if resume and ep == resume_epoch:
            activeloader = accelerator.skip_first_batches(trainloader, resume_step * accelerator.gradient_accumulation_steps)
        for batch in activeloader:
            with accelerator.accumulate(model):
                optimizer.zero_grad()
                output = model(**batch)
                loss = output.loss
                accelerator.backward(loss)
                optimizer.step()
                if accelerator.sync_gradients:
                    global_step += 1
                    if global_step % log_step == 0:
                        loss = accelerator.reduce(loss, reduction="mean")
                        accelerator.print(f"ep: {ep}, step: {global_step}, loss: {loss.item()}")
                        accelerator.log({"loss": loss.item()}, step=global_step)
                    if global_step % save_step == 0:
                        accelerator.save_state(accelerator.project_dir + f"/step_{global_step}")
                        accelerator.unwrap_model(model).save_pretrained(
                            save_directory=accelerator.project_dir + f"/step_{global_step}/model",
                            is_main_process=accelerator.is_main_process,
                            state_dict=accelerator.get_state_dict(model),
                            save_function=accelerator.save
                        )
        acc = evaluate(model, validloader, accelerator)
        accelerator.print(f"ep: {ep}, acc: {acc}")
        accelerator.log({"acc": acc}, step=global_step)
    accelerator.end_training()

def main():
    ## 混合精度训练 & 梯度累积 & 日志输出 & 模型保存 & 断点续训
    accelerator = Accelerator(
        mixed_precision="bf16", 
        gradient_accumulation_steps=2,
        log_with="tensorboard",
        project_dir="./accelerate",
    )
    accelerator.init_trackers("tf-logs")
    checkpoint = "./model/hfl/rbt3"
    model, optimizer = prepare_model_and_optimizer(checkpoint)
    trainloader, validloader = prepare_dataloader(checkpoint)
    model, optimizer, trainloader, validloader = accelerator.prepare(model, optimizer, trainloader, validloader)
    train(model, optimizer, trainloader, validloader, accelerator, resume=None)
    if dist.is_initialized():
        dist.destroy_process_group()

if __name__ == "__main__":
    main()