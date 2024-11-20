

### 环境

所需的关键 python 库如下：

-   accelerate== 1.0.1
-   bitsandbytes==0.44.1
-   datasets==2.18.0
-   ml-dtypes==0.4.1
-   numpy==1.26.4
-   optree==0.13.0
-   peft==0.7.1
-   pandas==2.0.3
-   safetensors==0.4.3
-   sentence-transformers2.2.2
-   sentencepiece0.2.0
-   torch==2.1.2+cu121
-   torchaudio==2.1.2+cu121
-   torchkeras==3.9.9
-   torchvision==0.16.2+cu121

另外 ，cuda 需要 12.1 版本。

### 硬件

1 张 4090 

### 模型

https://modelscope.cn/models/Qwen/Qwen2.5-0.5B

理论上从官方下载模型，仅需要这几个文件即可加载模型和分词器。

-   config.json
-   model.safetensors
-   tokenizer.json
-   tokenizer_config.json

### 数据

https://huggingface.co/datasets/hfl/ruozhiba_gpt4

弱智吧指令数据，样例展示如下，可以看出来这类数据的问题都比较弱智，但是答案是在用正常人的思维回答问题，示例如下:

```
{
  "instruction": "游泳比赛时把水喝光后跑步犯规吗",
  "input": "",
  "output": "在游泳比赛中，运动员的主要任务是完成泳道中的游泳距离，而不是喝水或跑步。因此，如果运动员在比赛中停止游泳而开始喝水甚至将水喝光，然后改为跑步，这显然违反了比赛的规则。游泳比赛有严格的规定，运动员必须在水中游泳规定的距离，并按照比赛的游泳技术要求来完成比赛。如果一个运动员选择在比赛过程中喝光水并开始跑步，这不仅违反了比赛规则，也是不现实的，因为实际上不可能在短时间内喝光泳池里的水。因此，这样的行为会被认为是犯规，运动员将被取消比赛资格。"
}
```

### 数据分析

这里是对数据的输入和输出 token 长度的关键分析代码，有助于我们后面设置模型的输入和输出长度。
```
import json
from transformers import AutoTokenizer
import numpy as np

def get_token_distribution(file_path, tokenizer):
    input_num_tokens, outout_num_tokens = [], []
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
        for obj in data:
            instruction = obj["instruction"]
            output = obj["output"]
            input_num_tokens.append(len(tokenizer(instruction).input_ids))
            outout_num_tokens.append(len(tokenizer(output).input_ids))
    return min(input_num_tokens), max(input_num_tokens), np.mean(input_num_tokens), np.percentile(input_num_tokens, 95), \
        min(outout_num_tokens), max(outout_num_tokens), np.mean(outout_num_tokens), np.percentile(outout_num_tokens, 95),

def main():
    model_path = "D:\Qwen2.5-0.5B-Instruct"
    train_data_path = "data/ruozhi/ruozhiba_train_2449.json"
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    i_min, i_max, i_avg, i_95, o_min, o_max, o_avg, o_95 = get_token_distribution(train_data_path, tokenizer)
    print(f"i_min：{i_min}, i_max：{i_max}, i_avg：{i_avg}, i_95:{i_95}, o_min：{o_min}, o_max：{o_max}, o_avg：{o_avg}, o_95:{o_95}")
```

数据输入和输出长度分析，i 表示输入， o 表示输出，95 表示百分之95分位数，最终我们选择取 i_95 和 o_95 的值：

```
i_min：1, i_max：72, i_avg：17.748060432829725, i_95:32.0, o_min：13, o_max：357, o_avg：104.60636994691711, o_95:151.5999999999999
```

### 微调方式

本文使用全参数指令微调的方式进行模型训练。

### 微调代码

```
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import time, sys

class NerDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_source_length, max_target_length) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.max_seq_length = self.max_source_length + self.max_target_length
        self.data = []
        if data_path:
            with open(data_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
                for obj in data:
                    instruction = obj["instruction"]
                    output = obj["output"]
                    self.data.append({
                        "instruction": instruction,
                        "output": output
                    })
        print("data load ， size：", len(self.data))

    def preprocess(self, instruction, output):
        messages = [ {"role": "system", "content": "你是一个有帮助的助手"}, {"role": "user", "content": instruction} ]
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        instruction = self.tokenizer(prompt, add_special_tokens=False, max_length=self.max_source_length, padding="max_length", pad_to_max_length=True, truncation=True)
        response = self.tokenizer(output, add_special_tokens=False, max_length=self.max_target_length, padding="max_length", pad_to_max_length=True, truncation=True)
        input_ids = instruction["input_ids"] + response["input_ids"] + [self.tokenizer.pad_token_id]
        attention_mask = (instruction["attention_mask"] + response["attention_mask"] + [1])
        labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [self.tokenizer.pad_token_id]
        return input_ids, attention_mask, labels

    def __getitem__(self, index):
        item_data = self.data[index]
        input_ids, attention_mask, labels = self.preprocess(**item_data)
        return { "input_ids": torch.LongTensor(np.array(input_ids)), "attention_mask": torch.LongTensor(np.array(attention_mask)), "labels": torch.LongTensor(np.array(labels)) }

    def __len__(self):
        return len(self.data)


def train_model(model, train_loader, val_loader, optimizer, device, num_epochs, model_output_dir, writer):
    batch_step = 0
    for epoch in range(num_epochs):
        time1 = time.time()
        model.train()
        for index, data in enumerate(tqdm(train_loader, file=sys.stdout, desc="Train Epoch: " + str(epoch))):
            input_ids = data['input_ids'].to(device, dtype=torch.long)
            attention_mask = data['attention_mask'].to(device, dtype=torch.long)
            labels = data['labels'].to(device, dtype=torch.long)
            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            writer.add_scalar('Loss/train', loss, batch_step)
            batch_step += 1
            # 100轮打印一次 loss
            if index % 100 == 0 or index == len(train_loader) - 1:
                time2 = time.time()
                tqdm.write( f"{index}, epoch: {epoch} -loss: {str(loss)} ; each step's time spent: {(str(float(time2 - time1) / float(index + 0.0001)))}")
        # 验证
        model.eval()
        val_loss = validate_model(model, device, val_loader)
        writer.add_scalar('Loss/val', val_loss, epoch)
        print(f"val loss: {val_loss} , epoch: {epoch}")
        print("Save Model To ", model_output_dir)
        model.save_pretrained(model_output_dir)


def validate_model(model, device, val_loader):
    running_loss = 0.0
    with torch.no_grad():
        for _, data in enumerate(tqdm(val_loader, file=sys.stdout, desc="Validation Data")):
            input_ids = data['input_ids'].to(device, dtype=torch.long)
            attention_mask = data['attention_mask'].to(device, dtype=torch.long)
            labels = data['labels'].to(device, dtype=torch.long)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, )
            loss = outputs.loss
            running_loss += loss.item()
    return running_loss / len(val_loader)


def main():
    model_name = "D:\Qwen2.5-0.5B-Instruct"
    train_json_path = "data/ruozhi/ruozhiba_train_2449.json"
    val_json_path = "data/ruozhi/ruozhiba_val_153.json"
    max_source_length = 62  # text 样本 i_95 是 32 ，模板自身大约 30 个，总共 62 个
    max_target_length = 151  # label 样本 o_95 是 151
    epochs = 3
    batch_size = 12
    lr = 1e-4
    model_output_dir = "sft_ruozhi"
    logs_dir = "logs"
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
    
    print("Start Load Train Data...")
    training_set = NerDataset(train_json_path, tokenizer, max_source_length, max_target_length)
    training_loader = DataLoader(training_set, batch_size=batch_size, shuffle=True, num_workers=4,)
    print("Start Load Validation Data...")
    val_set = NerDataset(val_json_path, tokenizer, max_source_length, max_target_length)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=4)
    
    writer = SummaryWriter(logs_dir)
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=lr)
    model = model.to(device)
    print("Start Training...")
    train_model( model=model, train_loader=training_loader, val_loader=val_loader, optimizer=optimizer, device=device, num_epochs=epochs, model_output_dir=model_output_dir, writer=writer )
    writer.close()

if __name__ == '__main__':
    main()
```

加载模型 `2.3G` ，训练需要 `18.4G` ，总共需要显存 `21.7G` ,训练和验证耗时总共 `5 分钟`左右。

### 微调之前的效果

微调之前，可以明显看出来，回答不符合弱智吧数据的风格，第一个甚至有乱码，第二个回答比问题还弱智，第三个回答脱离了主题：

```
user:地球上有70亿人，为什么只有86w人关注弱智吧，难道弱智这么少吗
assis:这是一个有趣的假设问题。实际上，地球上的人口数量是巨大的，而且每个人都有可能关注到弱智。弱智是指智力水平低于正常水平的人，他们可能在某些方面表现出色，但整体上可能无法被社会普遍接受或认可。因此，即使有8600000000000000000000000000000
user:游泳比赛时把水喝光后跑步犯规吗
assis:在游泳比赛中，如果运动员在比赛过程中把水喝光，这通常被视为违反比赛规则的行为，可能会导致比赛暂停或取消。这种行为可能被视为故意破坏比赛环境，违反了比赛规则，因此在某些情况下，可能会被取消比赛资格或处罚。

在实际比赛中，运动员通常会遵守比赛规则，尊重裁判和比赛环境。如果在比赛过程中出现这种情况，裁判可能会根据具体情况做出相应的处罚，包括取消比赛资格、警告等。因此，如果在游泳比赛中出现这种情况，运动员应该尊重裁判和比赛规则，避免任何可能的违规行为。
user:一些高中想提高升学率，为什么不直接招大学生？
assis:高中提高升学率，可以考虑以下几种方式：

1. **多样化课程设置**：提供更多的学科和课程，以满足不同学生的需求和兴趣。

2. **个性化教学**：采用个性化教学方法，根据学生的学习进度和能力进行调整。

3. **校本课程**：开设一些与学生兴趣相关的校本课程，如艺术、体育、科技等。

```

### 微调之后的效果

测试代码：

```
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def main():
    model_path = "D:\Qwen2.5-0.5B-Instruct"
    train_model_path = "sft_ruozhi"
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(train_model_path, trust_remote_code=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    test_case = [
        "地球上有70亿人，为什么只有86w人关注弱智吧，难道弱智这么少吗",
        "游泳比赛时把水喝光后跑步犯规吗",
        "一些高中想提高升学率，为什么不直接招大学生？"
    ]

    for case in test_case:
        messages = [ {"role": "system", "content": "你是一个有帮助的助手"}, {"role": "user", "content": case}]
        text = tokenizer.apply_chat_template( messages, tokenize=False, add_generation_prompt=True)
        model_inputs = tokenizer([text], return_tensors="pt").to(device)
        generated_ids = model.generate( model_inputs.input_ids, max_new_tokens=151, top_k=1)
        generated_ids = [ output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        print("----------------------------------")
        print(f"input: {case}\nresult: {response}")
```

微调之后进行推理，消耗显存 `2.4G` ，可以明显看出来回答更加符合原来的弱智吧问答数据的风格，问题虽然弱智，但是给出了比微调之前稍微合理的回答，尽管还是有瑕疵，这应该是数据量太少或者模型规模太小的原因，如下：

```
----------------------------------
input: 地球上有70亿人，为什么只有86w人关注弱智吧，难道弱智这么少吗
result: 关注“弱智吧”关注率是指某一群体对某一问题或事件的讨论和关注程度。关注率不能简单地用人数来判断，而是基于互联网使用习惯、经济发展、文化素养等多个因素。即使有大量关注“弱智吧”的人，也可能因为缺乏相关知识或信息，而不会关注或讨论与智力水平较低相关的内容。
----------------------------------
input: 游泳比赛时把水喝光后跑步犯规吗
result: 在游泳比赛中，运动员必须保持充足的水分摄入，比赛用的水体是流动的，运动员必须从水中不断跳入和跳出水体。因此，将水喝光并用跑步来代替实际的游泳动作是不成立的。游泳是一项需要身体及精神高度配合的运动，运动员必须在比赛过程中不断保持冷静并按照比赛的游泳技术要求来比赛。如果在比赛过程中停止游泳而开始用跑步代替游泳，这不仅违反了比赛的公平和专业道德，也是不现实的。
----------------------------------
input: 一些高中想提高升学率，为什么不直接招大学生？
result: 高中的主要使命是中学生中奖，而直接招收已经成为大学生的学生是完全不合适的。高中的存在价值在于帮助中学生完成从初中到大学的过渡，并且通过中奖来激励学生。如果直接招收已经成为大学生的学生，这将违背高中教育根本目的。高中的存在意义在于中奖，而直接招收已经成为大学生的学生，将无法实现这一目标。高中的存在价值在于帮助中学生完成从初中到大学的过渡，并通过中奖激励他们学习。如果直接招收已经成为大学生的学生，这将无法实现，因为高中的使命是中奖，而直接招收已经成为大学生的学生是不可能实现的。
```
### 结论

`说明 Qwen2.5-0.5B-Instruct 使用全参数指令微调的方式进行模型训练能够达到一定的下游任务预期效果。`