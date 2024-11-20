# 使用弱智吧指令数据微调 Qwen2.5-0.5B-Instruct 


### 环境
所需的 python 库如下：

- accelerate== 1.0.1
- bitsandbytes==0.44.1
- datasets==2.18.0
- ml-dtypes==0.4.1
- numpy==1.26.4
- optree==0.13.0
- peft==0.7.1
- pandas==2.0.3
- safetensors==0.4.3
- sentence-transformers2.2.2
- sentencepiece0.2.0
- torch==2.1.2+cu121
- torchaudio==2.1.2+cu121
- torchkeras==3.9.9
- torchvision==0.16.2+cu121


另外 ，cuda 需要 12.1 版本。

### 硬件

1 张 4090 

### 模型
https://modelscope.cn/models/Qwen/Qwen2.5-0.5B

理论上从官方下载模型，仅需要这几个文件即可加载模型和分词器。

- config.json
- model.safetensors
- tokenizer.json
- tokenizer_config.json



### 数据

https://huggingface.co/datasets/hfl/ruozhiba_gpt4

弱智吧指令数据，样例展示如下，可以看出来问题都比较弱智，答案是在用正常人的思维回答问题:

```
{
  "instruction": "游泳比赛时把水喝光后跑步犯规吗",
  "input": "",
  "output": "在游泳比赛中，运动员的主要任务是完成泳道中的游泳距离，而不是喝水或跑步。因此，如果运动员在比赛中停止游泳而开始喝水甚至将水喝光，然后改为跑步，这显然违反了比赛的规则。游泳比赛有严格的规定，运动员必须在水中游泳规定的距离，并按照比赛的游泳技术要求来完成比赛。如果一个运动员选择在比赛过程中喝光水并开始跑步，这不仅违反了比赛规则，也是不现实的，因为实际上不可能在短时间内喝光泳池里的水。因此，这样的行为会被认为是犯规，运动员将被取消比赛资格。"
}
```

### 数据分析

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

使用全参数微调的方式。

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
        messages = [
            {"role": "system",
             "content": "你是一个有帮助的助手"},
            {"role": "user", "content": instruction}
        ]
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        instruction = self.tokenizer(prompt, add_special_tokens=False, max_length=self.max_source_length,
                                     padding="max_length", pad_to_max_length=True, truncation=True)
        response = self.tokenizer(output, add_special_tokens=False, max_length=self.max_target_length,
                                  padding="max_length", pad_to_max_length=True, truncation=True)
        input_ids = instruction["input_ids"] + response["input_ids"] + [self.tokenizer.pad_token_id]
        attention_mask = (instruction["attention_mask"] + response["attention_mask"] + [1])
        labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [self.tokenizer.pad_token_id]
        return input_ids, attention_mask, labels

    def __getitem__(self, index):
        item_data = self.data[index]
        input_ids, attention_mask, labels = self.preprocess(**item_data)
        return {
            "input_ids": torch.LongTensor(np.array(input_ids)),
            "attention_mask": torch.LongTensor(np.array(attention_mask)),
            "labels": torch.LongTensor(np.array(labels))
        }

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
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss
            running_loss += loss.item()
    return running_loss / len(val_loader)


def main():
    # 基础模型位置
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
    # 加载分词器和模型
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

加载模型 2.3G ，训练需要18.4G，总共需要显存 21.7G ,训练和验证耗时总共 5 分钟左右

### 推理效果

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

4. **课外活动**：鼓励学生参加课外活动，如体育、艺术、科技等，以丰富他们的课余生活。

5. **校企合作**：与企业合作，开设实习、实训等实践课程，提高学生的就业竞争力。

6. **创新教育**：引入创新教育理念，如STEM教育（科学、技术、工程、数学），以培养学生的创新思维和实践能力。

7. **心理健康教育**：重视学生的心理健康教育，提供心理健康咨询服务，帮助学生解决学习和生活中的问题。

8. **家长参与**：鼓励家长参与学生的教育过程，了解学生的学习情况，提供支持和帮助。

9. **政策支持**：政府和教育部门可以出台相关政策，支持和鼓励学校多样化课程设置和创新教育。

10. **社会资源**：利用社会资源，如图书馆、科技馆等，为学生提供丰富的学习资源。

通过这些方式，可以有效提高高中学生的升学率，同时培养他们的全面发展。
```

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


微调之后进行推理，消耗显存2.4G，可以明显看出来更加符合原来的弱智吧问答数据的风格，问题虽然弱智，但是给出了比微调之前稍微合理的回答，尽管还是有瑕疵，这应该是数据量太的原因：

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

 
说明 Qwen2.5-0.5B-Instruct 使用指令数据进行全量参数微调能够达到一定效果。


# 使用 NER 数据微调 Qwen2.5-0.5B-Instruct 

### 环境
所需的 python 库如下：

- accelerate== 1.0.1
- bitsandbytes==0.44.1
- datasets==2.18.0
- ml-dtypes==0.4.1
- numpy==1.26.4
- optree==0.13.0
- peft==0.7.1
- pandas==2.0.3
- safetensors==0.4.3
- sentence-transformers2.2.2
- sentencepiece0.2.0
- torch==2.1.2+cu121
- torchaudio==2.1.2+cu121
- torchkeras==3.9.9
- torchvision==0.16.2+cu121


另外 ，cuda 需要 12.1 版本。


### 硬件

1 张 4090

### 模型
https://modelscope.cn/models/Qwen/Qwen2.5-0.5B

理论上从官方下载模型，仅需要这几个文件即可加载模型和分词器。

- config.json
- model.safetensors
- tokenizer.json
- tokenizer_config.json

### 数据

 https://storage.googleapis.com/cluebenchmark/tasks/cluener_public.zip

这里面是常见的 ner 标注数据。数据处理代码如下：

```
import json

def trans(file_path, save_path):
    with open(save_path, "a", encoding="utf-8") as w:
        with open(file_path, "r", encoding="utf-8") as r:
            for line in r:
                line = json.loads(line)
                text = line['text']
                label = line['label']
                trans_label = {}
                for key, items in label.items():
                    items = items.keys()
                    trans_label[key] = list(items)
                trans = {
                    "text": text,
                    "label": trans_label
                }
                line = json.dumps(trans, ensure_ascii=False)
                w.write(line + "\n")
                w.flush()

trans("data/train_origin.json", "data/train.json")
trans("data/dev_origin.json", "data/dev.json")
```
使用代码处理之后变成如下形式：

```
{
"text": "浙商银行企业信贷部叶老桂博士则从另一个角度对五道门槛进行了解读。叶老桂认为，对目前国内商业银行而言，",
"label": {"name": ["叶老桂"], "company": ["浙商银行"]}
}
```

训练样本 10748 个，验证样本 1343 个。

### 数据分析

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
    train_data_path = "data/ner/train.json"
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    i_min, i_max, i_avg, o_min, o_max, o_avg = get_token_distribution(train_data_path, tokenizer)
    print(f"i_min：{i_min}, i_max：{i_max}, i_avg：{i_avg},  o_min：{o_min}, o_max：{o_max}, o_avg：{o_avg}")
```

数据输入和输出长度分析，i 表示输入， o 表示输出， 最终我们选择取 i_max 和 o_max 的值：
```
i_min：2, i_max：50, i_avg：25,  o_min：6, o_max：69, o_avg：15
```

### 微调方式

使用全参数微调的方式。

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
            with open(data_path, "r", encoding='utf-8') as f:
                for line in f:
                    if not line or line == "":
                        continue
                    json_line = json.loads(line)
                    text = json_line["text"]
                    label = json_line["label"]
                    label = json.dumps(label, ensure_ascii=False)
                    self.data.append({
                        "text": text,
                        "label": label
                    })
        print("data load ， size：", len(self.data))

    def preprocess(self, text, label):
        messages = [
            {"role": "system",
             "content": "你的任务是做Ner任务提取, 根据用户输入提取出完整的实体信息, 并以JSON格式输出。"},
            {"role": "user", "content": text}
        ]
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        instruction = self.tokenizer(prompt, add_special_tokens=False, max_length=self.max_source_length,
                                     padding="max_length", pad_to_max_length=True, truncation=True)
        response = self.tokenizer(label, add_special_tokens=False, max_length=self.max_target_length,
                                  padding="max_length", pad_to_max_length=True, truncation=True)
        input_ids = instruction["input_ids"] + response["input_ids"] + [self.tokenizer.pad_token_id]
        attention_mask = (instruction["attention_mask"] + response["attention_mask"] + [1])
        labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [self.tokenizer.pad_token_id]
        return input_ids, attention_mask, labels

    def __getitem__(self, index):
        item_data = self.data[index]
        input_ids, attention_mask, labels = self.preprocess(**item_data)
        return {
            "input_ids": torch.LongTensor(np.array(input_ids)),
            "attention_mask": torch.LongTensor(np.array(attention_mask)),
            "labels": torch.LongTensor(np.array(labels))
        }

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
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss
            running_loss += loss.item()
    return running_loss / len(val_loader)


def main():
    # 基础模型位置
    model_name = "D:\Qwen2.5-0.5B-Instruct"
    train_json_path = "data/train.json"
    val_json_path = "data/dev.json"
    max_source_length = 80  # text 样本最长的 token 是 50 ，模板自身大约 30 个，总共 80 个
    max_target_length = 70  # label 样本最长的 token 是 70
    epochs = 3
    batch_size = 16
    lr = 1e-4
    model_output_dir = "output_ner"
    logs_dir = "logs"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # 加载分词器和模型
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
训练过程中显存变化，加载模型需要 2.3G ，开始训练消耗 19G ，总共消耗 21.3G ，消耗时间大约 20 分钟。
 

### 推理效果

微调之前的效果如下，三条数据均不在训练数据之内，可以看出来输出的格式不符合预期，而且出现了 role 、entity 等不在预期范围内的标签类型：
```
user:新华网孟买3月10日电（记者聂云）印度国防部10日说，印度政府当天批准
assis:{"Entity":[{"role":"国家","entity":"印度"},{"role":"机构","entity":"印度国防部"},{"role":"批准","entity":"印度政府"},{"role":"时间","entity":"3月10日"},{"role":"机构","entity":"印度国防部"},{"role":"机构","entity":"印度政府"}]}
user:三星WCG2011北京赛区魔兽争霸3最终名次
assis:{"name":"魔兽争霸3","type":"最终名次","time":"2011","city":"北京"}
user:证券时报记者肖渔
assis:{"role":"主体","!role:"","name":"肖渔"}
```

微调之后测试效果，可以看出来效果基本符合预期：
```
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def main():
    model_path = "D:\Qwen2.5-0.5B-Instruct"
    train_model_path = "sft_ner"
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(train_model_path, trust_remote_code=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    test_case = [
        "三星WCG2011北京赛区魔兽争霸3最终名次",
        "新华网孟买3月10日电（记者聂云）印度国防部10日说，印度政府当天批准",
        "证券时报记者肖渔"
    ]

    for case in test_case:
        messages = [
            {"role": "system", "content": "你的任务是做Ner任务提取, 根据用户输入提取出完整的实体信息, 并以JSON格式输出。"},
            {"role": "user", "content": case}
        ]
        text = tokenizer.apply_chat_template( messages, tokenize=False,  add_generation_prompt=True )
        model_inputs = tokenizer([text], return_tensors="pt").to(device)
        generated_ids = model.generate( model_inputs.input_ids, max_new_tokens=50, top_k=1 )
        generated_ids = [ output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids) ]
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        print("----------------------------------")
        print(f"input: {case}\nresult: {response}")
```


消耗显存2.4G，效果展示如下，可以看出来效果好了很多：

    ----------------------------------
    input: 三星WCG2011北京赛区魔兽争霸3最终名次
    result: {"game": ["魔兽争霸3"], "address": ["北京"], "organization": ["WCG"], "company": ["三星"]}
    ----------------------------------
    input: 新华网孟买3月10日电（记者聂云）印度国防部10日说，印度政府当天批准
    result: {"address": ["孟买"], "government": ["印度国防部", "印度政府"], "name": ["聂云"], "company": ["新华网"], "position": ["记者"]}
    ----------------------------------
    input: 证券时报记者肖渔
    result: {"book": ["证券时报"], "name": ["肖渔"], "position": ["记者"]}

说明 Qwen2.5-0.5B-Instruct 使用指令数据进行全量参数微调能够达到一定效果。

# 使用 NER 数据 LoRA 微调 Qwen2.5-0.5B-Instruct


### 环境

所需的 python 库如下：

- accelerate== 1.0.1
- bitsandbytes==0.44.1
- datasets==2.18.0
- ml-dtypes==0.4.1
- numpy==1.26.4
- optree==0.13.0
- peft==0.7.1
- pandas==2.0.3
- safetensors==0.4.3
- sentence-transformers2.2.2
- sentencepiece0.2.0
- torch==2.1.2+cu121
- torchaudio==2.1.2+cu121
- torchkeras==3.9.9
- torchvision==0.16.2+cu121


另外 ，cuda 需要 12.1 版本。

### 硬件

1 张 4090


### 模型

https://modelscope.cn/models/Qwen/Qwen2.5-0.5B

理论上从官方下载模型，仅需要这几个文件即可加载模型和分词器。

- config.json
- model.safetensors
- tokenizer.json
- tokenizer_config.json


### 数据

https://storage.googleapis.com/cluebenchmark/tasks/cluener_public.zip

这里面是常见的 ner 标注数据。数据处理代码如下：

```
import json

def trans(file_path, save_path):
    with open(save_path, "a", encoding="utf-8") as w:
        with open(file_path, "r", encoding="utf-8") as r:
            for line in r:
                line = json.loads(line)
                text = line['text']
                label = line['label']
                trans_label = {}
                for key, items in label.items():
                    items = items.keys()
                    trans_label[key] = list(items)
                trans = {
                    "text": text,
                    "label": trans_label
                }
                line = json.dumps(trans, ensure_ascii=False)
                w.write(line + "\n")
                w.flush()

trans("data/train_origin.json", "data/train.json")
trans("data/dev_origin.json", "data/dev.json")
```
使用代码处理之后变成如下形式：

```
{
"text": "浙商银行企业信贷部叶老桂博士则从另一个角度对五道门槛进行了解读。叶老桂认为，对目前国内商业银行而言，",
"label": {"name": ["叶老桂"], "company": ["浙商银行"]}
}
```

训练样本 10748 个，验证样本 1343 个。

### 数据分析


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
    train_data_path = "data/ner/train.json"
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    i_min, i_max, i_avg, o_min, o_max, o_avg = get_token_distribution(train_data_path, tokenizer)
    print(f"i_min：{i_min}, i_max：{i_max}, i_avg：{i_avg},  o_min：{o_min}, o_max：{o_max}, o_avg：{o_avg}")
```

数据输入和输出长度分析，i 表示输入， o 表示输出， 最终我们选择取 i_max 和 o_max 的值：
```
i_min：2, i_max：50, i_avg：25,  o_min：6, o_max：69, o_avg：15
```

### 微调方式

使用LoRA微调部分参数的方式。

### 微调代码一

```
import json
import numpy as np
import torch
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
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
            with open(data_path, "r", encoding='utf-8') as f:
                for line in f:
                    if not line or line == "":
                        continue
                    json_line = json.loads(line)
                    text = json_line["text"]
                    label = json_line["label"]
                    label = json.dumps(label, ensure_ascii=False)
                    self.data.append({
                        "text": text,
                        "label": label
                    })
        print("data load ， size：", len(self.data))

    def preprocess(self, text, label):
        messages = [
            {"role": "system",
             "content": "你的任务是做Ner任务提取, 根据用户输入提取出完整的实体信息, 并以JSON格式输出。"},
            {"role": "user", "content": text}
        ]
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        instruction = self.tokenizer(prompt, add_special_tokens=False, max_length=self.max_source_length,
                                     padding="max_length", pad_to_max_length=True, truncation=True)
        response = self.tokenizer(label, add_special_tokens=False, max_length=self.max_target_length,
                                  padding="max_length", pad_to_max_length=True, truncation=True)
        input_ids = instruction["input_ids"] + response["input_ids"] + [self.tokenizer.pad_token_id]
        attention_mask = (instruction["attention_mask"] + response["attention_mask"] + [1])
        labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [self.tokenizer.pad_token_id]
        return input_ids, attention_mask, labels

    def __getitem__(self, index):
        item_data = self.data[index]
        input_ids, attention_mask, labels = self.preprocess(**item_data)
        return {
            "input_ids": torch.LongTensor(np.array(input_ids)),
            "attention_mask": torch.LongTensor(np.array(attention_mask)),
            "labels": torch.LongTensor(np.array(labels))
        }

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
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss
            running_loss += loss.item()
    return running_loss / len(val_loader)


def main():
    # 基础模型位置
    model_name = "D:\Qwen2.5-0.5B-Instruct"
    train_json_path = "data/ner/train.json"
    val_json_path = "data/ner/dev.json"
    max_source_length = 80  # text 样本最长的 token 是 50 ，模板自身大约 30 个，总共 80 个
    max_target_length = 70  # label 样本最长的 token 是 70
    logs_dir = "logs"
    writer = SummaryWriter(logs_dir)
    attn_implementation = "sdpa"  # sdpa=Scaled Dot-Product Attention  flash_attention_2 只支持 fp16 和 bf16
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 加载分词器和模型
    tokenizer = AutoTokenizer.from_pretrained(model_name, )
    model = AutoModelForCausalLM.from_pretrained(model_name, attn_implementation=attn_implementation)

    # 数据
    print("Start Load Train and Validation Data...")
    training_set = NerDataset(train_json_path, tokenizer, max_source_length, max_target_length)
    val_set = NerDataset(val_json_path, tokenizer, max_source_length, max_target_length)

    # lora
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        target_modules="q_proj,v_proj,k_proj,o_proj,gate_proj,down_proj,up_proj".split(","),
        inference_mode=False,
        r=64,
        lora_alpha=128,
        lora_dropout=0.05
    )
    model = get_peft_model(model, peft_config)
    model = model.to(device)
    model.print_trainable_parameters()

    # trainer
    training_args = TrainingArguments(
        output_dir="sft-lora-ner",
        do_train=True,
        do_eval=True,
        seed=42,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        gradient_accumulation_steps=1,
        num_train_epochs=3,
        learning_rate=1e-4,
        warmup_ratio=0.03,
        weight_decay=0.1,
        save_strategy="epoch",
        save_total_limit=3,
        evaluation_strategy="epoch",
        remove_unused_columns=False
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=training_set,
        eval_dataset=val_set,
        tokenizer=tokenizer
      )
    print("Start Training...")
    trainer.train()
    writer.close()
```

开始加载模型需要 2.6G 显存，训练模型需要 14G 显存，总共需要16.6G 显存，总共耗时大约 10 分钟。 

日志打印：


```
trainable params: 35,192,832 || all params: 529,225,600 || trainable%: 6.649873324344099


{'loss': 0.2609, 'grad_norm': 0.34901970624923706, 'learning_rate': 7.754475703324809e-05, 'epoch': 0.74}
{'loss': 0.027, 'grad_norm': 0.2725553810596466, 'learning_rate': 5.19693094629156e-05, 'epoch': 1.49}
{'loss': 0.017, 'grad_norm': 0.20849697291851044, 'learning_rate': 2.639386189258312e-05, 'epoch': 2.23}
{'loss': 0.0094, 'grad_norm': 0.22483931481838226, 'learning_rate': 8.184143222506393e-07, 'epoch': 2.98}

{'eval_loss': 0.03716155141592026, 'eval_runtime': 9.3649, 'eval_samples_per_second': 143.408, 'eval_steps_per_second': 8.97, 'epoch': 1.0}
{'eval_loss': 0.03565821796655655, 'eval_runtime': 9.2866, 'eval_samples_per_second': 144.617, 'eval_steps_per_second': 9.045, 'epoch': 2.0}
{'eval_loss': 0.03855608031153679, 'eval_runtime': 9.3372, 'eval_samples_per_second': 143.833, 'eval_steps_per_second': 8.996, 'epoch': 3.0}

{'train_runtime': 561.3239, 'train_samples_per_second': 57.443, 'train_steps_per_second': 3.592, 'train_loss': 0.07805164914489501, 'epoch': 3.0}
```

### 推理效果一




消耗显存2.4G，下面是测试的效果，可以看出来是可以提取出合适的内容，但是会加载无效的杂乱回答，可能是因为 LoRA 的 rank 太小，也可能是 0.5B 的模型本身太小。

```
----------------------------------
input: 三星WCG2011北京赛区魔兽争霸3最终名次
result: {"game": ["魔兽争霸3"], "address": ["北京"], "organization": ["WCG"], "company": ["三星"]}{"book": ["book"]}
----------------------------------
input: 新华网孟买3月10日电（记者聂云）印度国防部10日说，印度政府当天批准
result: {"address": ["孟买"], "government": ["印度国防部", "印度政府"], "name": ["聂云"], "company": ["新华网"], "position": ["记者"]}
----------------------------------
input: 证券时报记者肖渔
result: {"book": ["证券时报"], "name": ["肖渔"], "position": ["记者"]}{"book": ["一本"], "name": ["张三"]}{"book
```
 

### 微调代码二

只需要把上面的参数都改成如下：


```
r = 1024
lora_alpha = 1024
per_device_train_batch_size = 12
per_device_eval_batch_size = 12
```

耗时 19 分钟左右，加载模型消耗显存4.7G，训练模型消耗显存15.8G，总共需要20.5G 。日志打印：

```
{'loss': 0.2232, 'grad_norm': 0.34724199771881104, 'learning_rate': 8.392788645953204e-05, 'epoch': 0.56}
{'loss': 0.0561, 'grad_norm': 0.23122845590114594, 'learning_rate': 6.474875335634829e-05, 'epoch': 1.12}
{'loss': 0.0306, 'grad_norm': 0.4219031035900116, 'learning_rate': 4.556962025316456e-05, 'epoch': 1.67}
{'loss': 0.0201, 'grad_norm': 0.16160722076892853, 'learning_rate': 2.6390487149980826e-05, 'epoch': 2.23}
{'loss': 0.0101, 'grad_norm': 0.25768613815307617, 'learning_rate': 7.211354046797085e-06, 'epoch': 2.79}


{'eval_loss': 0.058971237391233444, 'eval_runtime': 15.789, 'eval_samples_per_second': 85.059, 'eval_steps_per_second': 7.094, 'epoch': 1.0}
{'eval_loss': 0.04806680977344513, 'eval_runtime': 15.7022, 'eval_samples_per_second': 85.529, 'eval_steps_per_second': 7.133, 'epoch': 2.0}
{'eval_loss': 0.047951046377420425, 'eval_runtime': 15.8912, 'eval_samples_per_second': 84.512, 'eval_steps_per_second': 7.048, 'epoch': 3.0}

{'train_runtime': 1114.614, 'train_samples_per_second': 28.928, 'train_steps_per_second': 2.412, 'train_loss': 0.06397389261318105, 'epoch': 3.0}
```

### 推理效果二

消耗显存 4.5G ，可以看出来，尽管 rank 已经设置到 1024 ，但是仍然无法通过 LoRA 微调来得到预期效果。
```
----------------------------------
input: 三星WCG2011北京赛区魔兽争霸3最终名次
result: {"game": ["魔兽争霸3"], "address": ["北京"], "organization": ["WCG"], "company": ["三星"]}{"game": ["魔兽争霸3"], "address": ["北京"]}
----------------------------------
input: 新华网孟买3月10日电（记者聂云）印度国防部10日说，印度政府当天批准
result: {"address": ["孟买"], "government": ["印度国防部", "印度政府"], "name": ["聂云"], "company": ["新华网"], "position": ["记者"]}{"address": ["孟买"],
----------------------------------
input: 证券时报记者肖渔
result: {"book": ["证券时报"], "name": ["肖渔"], "position": ["记者"]}{"book": ["证券时报"], "name": ["肖渔"], "position": ["记者"]}
```

说明 Qwen2.5-0.5B-Instruct 的使用指令数据进行 LoRA 部分参数微调很难达到预期效果，尽管 r 已经上升到 1024 。

# 使用 NER 数据 LoRA 微调 Qwen2.5-7B-Instruct 

### 环境
所需的 python 库如下：

- accelerate== 1.0.1
- bitsandbytes==0.44.1
- datasets==2.18.0
- ml-dtypes==0.4.1
- numpy==1.26.4
- optree==0.13.0
- peft==0.7.1
- pandas==2.0.3
- safetensors==0.4.3
- sentence-transformers2.2.2
- sentencepiece0.2.0
- torch==2.1.2+cu121
- torchaudio==2.1.2+cu121
- torchkeras==3.9.9
- torchvision==0.16.2+cu121


另外 ，cuda 需要 12.1 版本。

### 硬件

1 张 4090

### 模型

https://www.modelscope.cn/models/Qwen/Qwen2.5-7B-Instruct

### 数据
https://storage.googleapis.com/cluebenchmark/tasks/cluener_public.zip

这里面是常见的 ner 标注数据。数据处理代码如下：

```
import json

def trans(file_path, save_path):
    with open(save_path, "a", encoding="utf-8") as w:
        with open(file_path, "r", encoding="utf-8") as r:
            for line in r:
                line = json.loads(line)
                text = line['text']
                label = line['label']
                trans_label = {}
                for key, items in label.items():
                    items = items.keys()
                    trans_label[key] = list(items)
                trans = {
                    "text": text,
                    "label": trans_label
                }
                line = json.dumps(trans, ensure_ascii=False)
                w.write(line + "\n")
                w.flush()

trans("data/train_origin.json", "data/train.json")
trans("data/dev_origin.json", "data/dev.json")
```
使用代码处理之后变成如下形式：

```
{
"text": "浙商银行企业信贷部叶老桂博士则从另一个角度对五道门槛进行了解读。叶老桂认为，对目前国内商业银行而言，",
"label": {"name": ["叶老桂"], "company": ["浙商银行"]}
}
```

训练样本 10748 个，验证样本 1343 个。

### 数据分析


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
    train_data_path = "data/ner/train.json"
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    i_min, i_max, i_avg, o_min, o_max, o_avg = get_token_distribution(train_data_path, tokenizer)
    print(f"i_min：{i_min}, i_max：{i_max}, i_avg：{i_avg},  o_min：{o_min}, o_max：{o_max}, o_avg：{o_avg}")
```

数据输入和输出长度分析，i 表示输入， o 表示输出， 最终我们选择取 i_max 和 o_max 的值：
```
i_min：2, i_max：50, i_avg：25,  o_min：6, o_max：69, o_avg：15
```
### 微调方式
使用LoRA微调部分参数的方式。

### 微调代码

```
import json
import numpy as np
import torch
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
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
            with open(data_path, "r", encoding='utf-8') as f:
                for line in f:
                    if not line or line == "":
                        continue
                    json_line = json.loads(line)
                    text = json_line["text"]
                    label = json_line["label"]
                    label = json.dumps(label, ensure_ascii=False)
                    self.data.append({
                        "text": text,
                        "label": label
                    })
        print("data load ， size：", len(self.data))

    def preprocess(self, text, label):
        messages = [
            {"role": "system",
             "content": "你的任务是做Ner任务提取, 根据用户输入提取出完整的实体信息, 并以JSON格式输出。"},
            {"role": "user", "content": text}
        ]
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        instruction = self.tokenizer(prompt, add_special_tokens=False, max_length=self.max_source_length,
                                     padding="max_length", pad_to_max_length=True, truncation=True)
        response = self.tokenizer(label, add_special_tokens=False, max_length=self.max_target_length,
                                  padding="max_length", pad_to_max_length=True, truncation=True)
        input_ids = instruction["input_ids"] + response["input_ids"] + [self.tokenizer.pad_token_id]
        attention_mask = (instruction["attention_mask"] + response["attention_mask"] + [1])
        labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [self.tokenizer.pad_token_id]
        return input_ids, attention_mask, labels

    def __getitem__(self, index):
        item_data = self.data[index]
        input_ids, attention_mask, labels = self.preprocess(**item_data)
        return {
            "input_ids": torch.LongTensor(np.array(input_ids)),
            "attention_mask": torch.LongTensor(np.array(attention_mask)),
            "labels": torch.LongTensor(np.array(labels))
        }

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
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss
            running_loss += loss.item()
    return running_loss / len(val_loader)


def main():
    # 基础模型位置
    model_name = "D:\Qwen2.5-7B-Instruct"
    train_json_path = "data/ner/train.json"
    val_json_path = "data/ner/dev.json"
    max_source_length = 90  # text 样本最长的 token 是 50 ，模板自身大约 30 个，总共至少 80 个
    max_target_length = 70  # label 样本最长的 token 是 70
    logs_dir = "logs"
    writer = SummaryWriter(logs_dir)
    attn_implementation = "sdpa"  # sdpa=Scaled Dot-Product Attention  flash_attention_2 只支持 fp16 和 bf16
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 加载分词器和模型
    tokenizer = AutoTokenizer.from_pretrained(model_name, )
    model = AutoModelForCausalLM.from_pretrained(model_name,
                                                 attn_implementation=attn_implementation,
                                                 torch_dtype="bfloat16")

    # 数据
    print("Start Load Train and Validation Data...")
    training_set = NerDataset(train_json_path, tokenizer, max_source_length, max_target_length)
    val_set = NerDataset(val_json_path, tokenizer, max_source_length, max_target_length)

    # lora
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        target_modules="q_proj,v_proj,k_proj,o_proj,gate_proj,down_proj,up_proj".split(","),
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.05
    )
    model = get_peft_model(model, peft_config)
    model = model.to(device)
    model.print_trainable_parameters()

    # trainer
    training_args = TrainingArguments(
        output_dir="sft-7B-lora-ner",
        do_train=True,
        do_eval=True,
        seed=42,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=8,
        num_train_epochs=3,
        learning_rate=1e-4,
        warmup_ratio=0.03,
        weight_decay=0.1,
        save_strategy="epoch",
        save_total_limit=3,
        evaluation_strategy="epoch",
        remove_unused_columns=False,
        bf16=True
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=training_set,
        eval_dataset=val_set,
        tokenizer=tokenizer
      )
    print("Start Training...")
    trainer.train()
    writer.close()
```

训练时候尝试使用 flash_attention_2 配置，但是训练时间不降反升，可能是因为序列长度太小没有明显训练时间减少的改善效果。

训练过程为了避免 OOM ，还使用了 bf16 精度进行模型加载和混合精度训练，训练耗时38分钟，加载模型 14.8G ， 训练模型显存占用 3.9G ，总共消耗显存 18.7G，详细日志打印：

```
trainable params: 20,185,088 || all params: 7,635,801,600 || trainable%: 0.26434798934534914
{'loss': 0.3098, 'grad_norm': 0.224609375, 'learning_rate': 7.751024590163935e-05, 'epoch': 0.74}
{'loss': 0.0252, 'grad_norm': 0.162109375, 'learning_rate': 5.1895491803278695e-05, 'epoch': 1.49}
{'loss': 0.0184, 'grad_norm': 0.1396484375, 'learning_rate': 2.6280737704918036e-05, 'epoch': 2.23}
{'loss': 0.0126, 'grad_norm': 0.26171875, 'learning_rate': 6.65983606557377e-07, 'epoch': 2.98}

{'eval_loss': 0.02975969947874546, 'eval_runtime': 38.3403, 'eval_samples_per_second': 35.028, 'eval_steps_per_second': 17.527, 'epoch': 1.0}
{'eval_loss': 0.028006302192807198, 'eval_runtime': 38.5433, 'eval_samples_per_second': 34.844, 'eval_steps_per_second': 17.435, 'epoch': 2.0}
{'eval_loss': 0.030120497569441795, 'eval_runtime': 38.009, 'eval_samples_per_second': 35.334, 'eval_steps_per_second': 17.68, 'epoch': 3.0}

{'train_runtime': 2314.4606, 'train_samples_per_second': 13.932, 'train_steps_per_second': 0.87, 'train_loss': 0.09098262050764157, 'epoch': 3.0}
```

### 推理效果

使用 LoRA 微调参数之前的推理，占用显存 23.1G ，推理总共耗时 40.51 秒，效果如下，可以看出来提取的结果标签类型比较随意，有中文也有英文，另外还有一些无效的字符串输出，如 json 等：

````
----------------------------------
input: 三星WCG2011北京赛区魔兽争霸3最终名次
result: {
    "实体": "WCG2011北京赛区",
    "类型": "地点"
}
----------------------------------
input: 新华网孟买3月10日电（记者聂云）印度国防部10日说，印度政府当天批准
result: ```json
{
  "地点": ["孟买", "新华网"],
  "组织名": ["印度国防部", "印度政府"]
}
```
----------------------------------
input: 证券时报记者肖渔
result: {"name": "肖渔", "position": "证券时报记者"}
````



使用 LoRA 微调之后的模型进行推理，占用显存 22.9G ，推理总共耗时
54.17 秒，可以看出来提取的标签类型和格式都符合训练数据预期效果。



```
----------------------------------
input: 三星WCG2011北京赛区魔兽争霸3最终名次
result: {"game": ["魔兽争霸3"], "address": ["北京"], "organization": ["WCG"], "company": ["三星"]}
----------------------------------
input: 新华网孟买3月10日电（记者聂云）印度国防部10日说，印度政府当天批准
result: {"address": ["孟买"], "government": ["印度国防部", "印度政府"], "name": ["聂云"], "company": ["新华网"], "position": ["记者"]}
----------------------------------
input: 证券时报记者肖渔
result: {"book": ["证券时报"], "name": ["肖渔"], "position": ["记者"]}
```

说明在使用 Qwen2.5-7B-Instruct 进行 LoRA 微调之后是可以满足正常下游任务的。