# 前文
最近一直做的 txt2sql 的项目，使用的大模型的是阿里的通义千问的最新的最大的模型 api 接口，但是领导在充了 2000 块之后觉得太贵了，一直花钱不是个事情，让我开始研究能不能微调下开源的 codellama 来替换通义千问，所以才有了下面的这些事情。我在网上找了个教程（见文末），学着微调，给大家展示下微调效果，效果一塌糊涂， 但是整体的思路没问题。

# 准备
工先欲其事，必先利其器，准备工作是少不了的。大体有以下的基本工作要做好：
- 选择要微调的模型，这里选择的是 `CodeLlama-13b-hf` ，因为是实验性质的所以搞个最方便微调的模型即可，直接从国内的 [modelscope](https://community.modelscope.cn/) 下载会快一点。（其实我就是没有设备，只能玩最小的，领导也不给配置，哭死...）
- 准备微调模型的样本数据，这个都是业务上积攒下来的，比较好获取
- 4090 显卡，没这个基本玩不起来
- 配置 torch 环境，这个网上很多，就不多说了


# 训练数据
训练数据的模板我是在网上找的，每条数据基本上就是按照下面的套路来的：

```
d = {
            "context":"",
            "question":"",
            "answer":""
        }
```

可以看出来每个样本是一个 json 格式的数据，里面有三个字段 `context`、`question`、`answer` ，分表标识的含义是上下文内容，问题，答案，我们这里因为要做的任务是 txt2sql ，所以 question 是用户的问题，answer 是提前准备好的能够解决用户问题的标准 sql ，context 是解决用户问题需要的数据库相关元数据信息。最后给大家展示一个样本示例：


```
{
'context': "\n\n您可以使用以下展示出的 DDL 作为参考，每个 DDL 描述了表的表名、字段名以及对应的类型，使用他们以指导您有效准确地回答用户的问题，请务必正确使用每个字段的名字和类型，禁止捏造虚假内容:\n\nCREATE TABLE gxssln\n(\n    id integer ,\n    gxid varying(100),\n ......  id varying(100),\n\tname text ,\n\tgeom geometry(Geometry,4326)\n)\n\n", 
'question': '杭州市长输管线管龄在20-30年的有多长', 
'answer': 'SELECT round(SUM(length::numeric)/1000, 2) as "长度总和(公里)" from gxssln WHERE pipeline_type LIKE '%GA%' AND gl >= 20 AND gl < 30\n'
}
```

# 微调之前的效果
我已经从 modelScope 社区中把 `CodeLlama-13b-hf` 的模型下载好了，然后我们在微调之前初步体验一下直接使用模型的效果。我这里提出了自己的业务问题 `查一下市民中心站附近500米的人防工程的数量` ，并且将自己的数据库表信息作为了上下文内容传入了。
```
import torch
from transformers import (AutoTokenizer, AutoModelForCausalLM)

# 读取模型
base_model =   r'D:\CodeLlama-13b-hf'

model = AutoModelForCausalLM.from_pretrained(
    base_model,
    load_in_8bit=True,
    torch_dtype=torch.float16,
    device_map="auto",
    low_cpu_mem_usage=True
)

tokenizer = AutoTokenizer.from_pretrained(base_model)
tokenizer.pad_token = tokenizer.eos_token
eval_prompt = """ 你是一个强大的文本到 SQL 模型。你的工作是回答有关数据库的问题。系统会为你提供一个问题和有关一个或多个表的上下文。 你必须输出回答该问题的 SQL 查询。最后只输出 SQL ，不输出其他内容。最后只输出 SQL ，不输出其他内容。


### Context:
CREATE TABLE rfgcpy
(
id varying(100),
name varying(100),
jldw varying(100),
qsdw varying(100),
jsskwsd numeric,
ssqx varying(100),
tyaqpjdj integer,
jgsj integer,
geom geometry(Geometry,4326),
)

CREATE TABLE aipoi
(
id varying(100),
name text ,
geom geometry(Geometry,4326)
)

 
### Question:
查一下市民中心站附近500米的人防工程的数量

### Answer:
"""
model_input = tokenizer(eval_prompt, return_tensors="pt").to("cuda")

model.eval()
with torch.no_grad():
    print(tokenizer.decode(model.generate(**model_input, max_new_tokens=100)[0], skip_special_tokens=True))
```
加载模型并推理的时候消耗 GPU 大概 17.1G 左右，结果输出：

```sql
SELECT count(*)
FROM rfgcpy
WHERE ST_DWithin(geom,ST_GeomFromText('POINT(116.390999 39.913999)',4326),500)
```
可以看出这个结果是错的，怎么经纬度都出来了，我都不知道经纬度是多少，基本是瞎蒙出来的。

# 微调




下面是我用 `peft` 库来进行的模型微调代码，如果大家想拿来直接用，直接修改`训练数据位置`、`需要微调的模型位置`、`微调后的保存位置`、`训练模型相关的超参数`（尤其是：`batch_size`，我的显存配置 `24G` ，也只能够设置为 `2`，微调需要占用 `23G` 左右的显存，再多一点我的 4090 都要爆 ）。

```
from datetime import datetime
import torch
from datasets import load_dataset
from peft import (LoraConfig, get_peft_model, prepare_model_for_int8_training, )
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForSeq2Seq

train_dataset = load_dataset('json', data_files='微调 code llama 训练数据 2.jsonl', split='train')
eval_dataset = load_dataset('json', data_files='微调 code llama 验证数据 2.jsonl', split='train')
print(train_dataset.shape, eval_dataset.shape)
base_model = r'D:\CodeLlama-13b-hf'     
model = AutoModelForCausalLM.from_pretrained( base_model,  load_in_8bit=True,  torch_dtype=torch.float16,  device_map="auto", )  
tokenizer = AutoTokenizer.from_pretrained(base_model)
tokenizer.add_eos_token = True
tokenizer.pad_token_id = 0
tokenizer.padding_side = "left"

def tokenize(prompt):
    result = tokenizer( prompt,  truncation=True,  max_length=512,  padding=False, return_tensors=None, )
    result["labels"] = result["input_ids"].copy()
    return result

def generate_and_tokenize_prompt(data_point):
    full_prompt =f"""您是一个强大的文本到 SQL 模型。您的工作是回答有关数据库的问题。您将获得有关关于问题的多个表作为上下文。 您必须输出回答该问题的 SQL 查询。请直接返回 sql ，不做任何解释。请直接返回 sql ，不做任何解释。请直接返回 sql ，不做任何解释。\n
                ### Context: \n{data_point["context"]}
                ### Question: \n{data_point["question"]}
                ### Answer: \n```sql\n{data_point["answer"]}``` """
    return tokenize(full_prompt)


tokenized_train_dataset = train_dataset.map(generate_and_tokenize_prompt)
tokenized_val_dataset = eval_dataset.map(generate_and_tokenize_prompt)
model.train()
model = prepare_model_for_int8_training(model)
config = LoraConfig( r=16, lora_alpha=16, target_modules=[ "q_proj", "k_proj", "v_proj", "o_proj", ],  lora_dropout=0.05, bias="none", task_type="CAUSAL_LM", )
model = get_peft_model(model, config)

batch_size = 2
per_device_train_batch_size = 2
gradient_accumulation_steps = batch_size // per_device_train_batch_size
output_dir = "D:\PycharmProjects\202406\code-llama-ft"
training_args = TrainingArguments( per_device_train_batch_size=per_device_train_batch_size, gradient_accumulation_steps=gradient_accumulation_steps,  warmup_steps=100, num_train_epochs=3,  learning_rate=3e-4,  fp16=True, logging_steps=100,  optim="adamw_torch",  evaluation_strategy="steps",  save_strategy="steps",  eval_steps=500,  save_steps=100,  output_dir=output_dir, load_best_model_at_end=False,  group_by_length=True,  report_to="none", run_name=f"codellama-{datetime.now().strftime('%Y-%m-%d-%H-%M')}", )
trainer = Trainer(   model=model,  train_dataset=tokenized_train_dataset, eval_dataset=tokenized_val_dataset, args=training_args, data_collator=DataCollatorForSeq2Seq( tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True  ), )
model.config.use_cache = False
trainer.train()
```

 直接使用这份代码是没问题的，但是运行的时候在保存模型的时候会报错：
 ```
 PermissionError: [Errno 13] Permission denied: 'D:\\PycharmProjects\\202406\\code-llama-ft\\checkpoint-20'
 ```
 我在网上查了很多资料，发现大家都遇到了这个问题，是源代码的问题，如下图直接将 `trainer.py` 的 `2417-2420 行`都注释就可以了。

![image.png](https://p6-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/e6ab5108e2b44475a4e3bdda4ef50481~tplv-k3u1fbpfcp-jj-mark:0:0:0:0:q75.image#?w=782&h=96&s=12133&e=png&b=202125)

日志打印（我把日志简单处理了一下，方便浏览）：
 ```
{'loss': 0.6034, 'learning_rate': 0.0003, 'epoch': 0.18}
{'loss': 0.0145, 'learning_rate': 0.0002803020354563362, 'epoch': 0.37}
{'loss': 0.005, 'learning_rate': 0.00026060407091267233, 'epoch': 0.55}
{'loss': 0.0061, 'learning_rate': 0.0002409061063690085, 'epoch': 0.74}
{'loss': 0.0047, 'learning_rate': 0.00022120814182534468, 'epoch': 0.92}
{'loss': 0.0045, 'learning_rate': 0.00020151017728168089, 'epoch': 1.11}
{'loss': 0.0044, 'learning_rate': 0.00018181221273801706, 'epoch': 1.29}
{'loss': 0.0064, 'learning_rate': 0.00016211424819435324, 'epoch': 1.48}
{'loss': 0.0055, 'learning_rate': 0.00014241628365068942, 'epoch': 1.66}
{'loss': 0.0056, 'learning_rate': 0.0001227183191070256, 'epoch': 1.85}
{'loss': 0.0043, 'learning_rate': 0.00010302035456336177, 'epoch': 2.03}
{'loss': 0.0041, 'learning_rate': 8.332239001969796e-05, 'epoch': 2.22}
{'loss': 0.0042, 'learning_rate': 6.362442547603414e-05, 'epoch': 2.4}
{'loss': 0.0042, 'learning_rate': 4.3926460932370315e-05, 'epoch': 2.59}
{'loss': 0.0054, 'learning_rate': 2.42284963887065e-05, 'epoch': 2.77}
{'loss': 0.0048, 'learning_rate': 4.530531845042678e-06, 'epoch': 2.96}
{'eval_loss': 0.004683102015405893, 'eval_runtime': 4.0643, 'eval_samples_per_second': 7.135, 'eval_steps_per_second': 0.984, 'epoch': 0.92}
{'eval_loss': 0.004700557328760624, 'eval_runtime': 4.0351, 'eval_samples_per_second': 7.187, 'eval_steps_per_second': 0.991, 'epoch': 1.85}
{'eval_loss': 0.0047799041494727135, 'eval_runtime': 4.035, 'eval_samples_per_second': 7.187, 'eval_steps_per_second': 0.991, 'epoch': 2.77}
{'train_runtime': 2489.3098, 'train_samples_per_second': 1.303, 'train_steps_per_second': 0.652, 'train_loss': 0.04239478127976956, 'epoch': 3.0}
 ```
可以看出来一共训练了 `3` 个 epoch ， 训练期间的 `loss` 和 验证期间的 `eval_loss` 也是在逐渐收敛，说明训练是有效的。

# 微调之后的效果

这里是我用微调之后的 lora 来测试的效果，耗用显存 17G 左右。

```
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base_model = r'D:\CodeLlama-13b-hf'   # 17.3G
model = AutoModelForCausalLM.from_pretrained( base_model, load_in_8bit=True,  torch_dtype=torch.float16, device_map="auto",)
tokenizer = AutoTokenizer.from_pretrained(base_model)
output_dir = r"D:\PycharmProjects\202406\code-llama-ft\checkpoint-1600"
model = PeftModel.from_pretrained(model, output_dir)
eval_prompt = """你是一个强大的文本到 SQL 模型。你的工作是回答有关数据库的问题。系统会为你提供一个问题和有关一个或多个表的上下文。 你必须输出回答该问题的 SQL 查询。最后只输出 SQL ，不输出其他内容。最后只输出 SQL ，不输出其他内容。


### Context:
CREATE TABLE rfgcpy
(
id varying(100),
name varying(100),
jldw varying(100),
qsdw varying(100),
jsskwsd numeric,
ssqx varying(100),
tyaqpjdj integer,
jgsj integer,
geom geometry(Geometry,4326),
)

CREATE TABLE aipoi
(
id varying(100),
name text ,
geom geometry(Geometry,4326)
)

 
### Question:
查一下市民中心站附近500米的人防工程的数量

### Answer:"""
model_input = tokenizer(eval_prompt, return_tensors="pt").to("cuda")
model.eval()
with torch.no_grad():
    outputs = model.generate(**model_input, max_new_tokens=100)[0]
print(tokenizer.decode(outputs, skip_special_tokens=True))
```


运行会报错：
```
safetensors_rust.SafetensorError: Error while deserializing header: InvalidHeaderDeserialization
```

如果微调代码有下面代码将其删除，重新微调即可，将微调的路径修改成和我一样到具体的 lora 模型位置就可以：
```
old_state_dict = model.state_dict
model.state_dict = (lambda self, *_, **__: get_peft_model_state_dict(self, old_state_dict())).__get__(
    model, type(model)
)
if torch.__version__ >= "2" and sys.platform != "win32":
    print("compiling the model")
    model = torch.compile(model)
```
结果打印：

```
SELECT COUNT(*) as "数量" FROM rfgcpy WHERE geom AND ssqx AND name AND jsdw AND jsskwsd AND tyaqpjdj AND jgsj is not null  AND geom AND ssqx AND name AND jsdw AND jsskwsd AND tyaqpjdj AND jgsj is not null  AND geom AND ssqx AND name AND jsdw AND jsskwsd AND tya
```

打印出来的东西简直狗屁不通，甚至都不完整，还是 TM 是个断句。我找前辈聊了一下，`基本上认定有两个原因，第一个就是可能中文支持不是很好，第二个原因就是微调的训练数据量太少，基本上要上一万才能行。至此我浪费了两天时间做了个不能用的垃圾出来，与君共勉。`


# 参考

- 微调教程：http://www.yxfzedu.com/article/5631
- 微调教程：https://github.com/ragntune/code-llama-finetune/blob/main/fine-tune-code-llama.ipynb
- 模型下载：https://www.modelscope.cn/models/AI-ModelScope/CodeLlama-13b-hf/files