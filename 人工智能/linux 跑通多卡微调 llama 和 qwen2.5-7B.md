 linux 跑通多卡微调 llama 和 qwen2.5-7B
 
 # 选卡和镜像
 
 
## 卡
 我这里选择的是4090卡，具体算力自己查看即可。

 
![image.png](https://p26-juejin-sign.byteimg.com/tos-cn-i-k3u1fbpfcp/94d942d160334816a364d7f20670ad67~tplv-k3u1fbpfcp-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6:q75.awebp?rk3s=bb34011d&x-expires=1731132075&x-signature=XM8nBMy3lZug1%2B5%2FjfvOt1hZsuE%3D)

 ## 镜像
 
 选择 autoDL 自带的镜像，方便一些。
 
![image.png](https://p26-juejin-sign.byteimg.com/tos-cn-i-k3u1fbpfcp/7c734bec79894c54afe49f7c43e1e317~tplv-k3u1fbpfcp-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6:q75.awebp?rk3s=bb34011d&x-expires=1731132075&x-signature=vAN9lKUvPv7HU%2B5e%2FlayI2%2FBiUM%3D)

如果你有自己的镜像，也可以用自己的镜像创建新的实例。
 
 # 准备工作
 
 可以先选择**无卡模型**进行环境配置，比较省钱一点，等需要跑模型的时候再用卡。
 
 
 ## 虚拟环境
 
- 创建：conda create --name torch-2.1-py-3.10 python=3.10  
- 进入：conda activate torch-2.1-py-3.10，如果报错如下，只需要彻底关闭当前 shell ，重启之后再执行命令即可。

        conda activate torch-2.1-py-3.10 

        CommandNotFoundError: Your shell has not been properly configured to use 'conda activate'.
        To initialize your shell, run

            $ conda init <SHELL_NAME>

        Currently supported shells are:
          - bash
          - fish
          - tcsh
          - xonsh
          - zsh
          - powershell

        See 'conda init --help' for more information and options.

        IMPORTANT: You may need to close and restart your shell after running 'conda init'.
  
  然后开始安装目标版本的 pytorch 和各种库，可能会有版本冲突，但是一个一个排查解决即可。不管有没有报错冲突，只要 pip list 看有成功装上即可。
  
```
-   transformers==4.46.0
-   optree==0.13.0
-   deepspeed==0.15.3
-   ml_dtypes==0.5.0(可选)
-   tf_keras==2.17.0(可选)
-   peft==0.7.1
-   accelerate==1.0.1
-   bitsandbytes==0.44.1
-   torch==2.1.2+cu121
-   torchaudio==2.1.2+cu121
-   torchkeras==3.9.9(可选)
-   torchvision==0.16.2+cu121
-   tensorboard==2.18.0
-   numpy==1.26.4
-   pandas==2.0.3
-   datasets==2.18.0
-   tokenizers==0.20.1
-   xformers==0.0.28.post2
-   dataclasses-json==0.6.4
```
  
 - 安装 flash-attn ，我下载的是 flash_attn-2.6.3+cu118torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl 
 - （可选，只要处理量化才需要）拉 llama.cpp 仓库，然后进入到文件目录之中进行 make 编译。会打印如下信息：
```
I ccache not found. Consider installing it for faster compilation.
I llama.cpp build info: 
I UNAME_S:   Linux
I UNAME_P:   x86_64
I UNAME_M:   x86_64
I CFLAGS:    -Iggml/include -Iggml/src -Iinclude -Isrc -Icommon -D_XOPEN_SOURCE=600 -D_GNU_SOURCE -DNDEBUG -DGGML_USE_OPENMP -DGGML_USE_LLAMAFILE -DGGML_USE_AMX  -std=c11   -fPIC -O3 -g -Wall -Wextra -Wpedantic -Wcast-qual -Wno-unused-function -Wshadow -Wstrict-prototypes -Wpointer-arith -Wmissing-prototypes -Werror=implicit-int -Werror=implicit-function-declaration -pthread -march=native -mtune=native -fopenmp -Wdouble-promotion 
I CXXFLAGS:  -std=c++11 -fPIC -O3 -g -Wall -Wextra -Wpedantic -Wcast-qual -Wno-unused-function -Wmissing-declarations -Wmissing-noreturn -pthread -fopenmp  -march=native -mtune=native -Wno-array-bounds -Wno-format-truncation -Wextra-semi -Iggml/include -Iggml/src -Iinclude -Isrc -Icommon -D_XOPEN_SOURCE=600 -D_GNU_SOURCE -DNDEBUG -DGGML_USE_OPENMP -DGGML_USE_LLAMAFILE -DGGML_USE_AMX 
I NVCCFLAGS: -std=c++11 -O3 -g 
I LDFLAGS:    
I CC:        cc (Ubuntu 11.3.0-1ubuntu1~22.04) 11.3.0
I CXX:       c++ (Ubuntu 11.3.0-1ubuntu1~22.04) 11.3.0

c++ -std=c++11 -fPIC -O3 -g -Wall -Wextra -Wpedantic -Wcast-qual -Wno-unused-function -Wmissing-declarations -Wmissing-noreturn -pthread -fopenmp  -march=native -mtune=native -Wno-array-bounds -Wno-format-truncation -Wextra-semi -Iggml/include -Iggml/src -Iinclude -Isrc -Icommon -D_XOPEN_SOURCE=600 -D_GNU_SOURCE -DNDEBUG -DGGML_USE_OPENMP -DGGML_USE_LLAMAFILE -DGGML_USE_AMX  -c ggml/src/llamafile/sgemm.cpp -o ggml/src/llamafile/sgemm.o
c++ -std=c++11 -fPIC -O3 -g -Wall -Wextra -Wpedantic -Wcast-qual -Wno-unused-function -Wmissing-declarations -Wmissing-noreturn -pthread -fopenmp  -march=native -mtune=native -Wno-array-bounds -Wno-format-truncation -Wextra-semi -Iggml/include -Iggml/src -Iinclude -Isrc -Icommon -D_XOPEN_SOURCE=600 -D_GNU_SOURCE -DNDEBUG -DGGML_USE_OPENMP -DGGML_USE_LLAMAFILE -DGGML_USE_AMX  -c ggml/src/ggml-amx.cpp -o ggml/src/ggml-amx.o
...省略
c++ -std=c++11 -fPIC -O3 -g -Wall -Wextra -Wpedantic -Wcast-qual -Wno-unused-function -Wmissing-declarations -Wmissing-noreturn -pthread -fopenmp  -march=native -mtune=native -Wno-array-bounds -Wno-format-truncation -Wextra-semi -Iggml/include -Iggml/src -Iinclude -Isrc -Icommon -D_XOPEN_SOURCE=600 -D_GNU_SOURCE -DNDEBUG -DGGML_USE_OPENMP -DGGML_USE_LLAMAFILE -DGGML_USE_AMX  ggml/src/llamafile/sgemm.o ggml/src/ggml-amx.o ggml/src/ggml-amx/mmq.o ggml/src/ggml.o ggml/src/ggml-alloc.o ggml/src/ggml-backend.o ggml/src/ggml-quants.o ggml/src/ggml-aarch64.o src/llama.o src/llama-vocab.o src/llama-grammar.o src/llama-sampling.o src/unicode.o src/unicode-data.o common/common.o common/arg.o common/log.o common/console.o common/ngram-cache.o common/sampling.o common/train.o common/build-info.o common/json-schema-to-grammar.o examples/main/main.o -o llama-cli  

====  Run ./llama-cli -h for help.  ====

c++ -std=c++11 -fPIC -O3 -g -Wall -Wextra -Wpedantic -Wcast-qual -Wno-unused-function -Wmissing-declarations -Wmissing-noreturn -pthread -fopenmp  -march=native -mtune=native -Wno-array-bounds -Wno-format-truncation -Wextra-semi -Iggml/include -Iggml/src -Iinclude -Isrc -Icommon -D_XOPEN_SOURCE=600 -D_GNU_SOURCE -DNDEBUG -DGGML_USE_OPENMP -DGGML_USE_LLAMAFILE -DGGML_USE_AMX  -c examples/convert-llama2c-to-ggml/convert-llama2c-to-ggml.cpp -o examples/convert-llama2c-to-ggml/convert-llama2c-to-ggml.o
...省略
NOTICE: The 'main' binary is deprecated. Please use 'llama-cli' instead.
c++ -std=c++11 -fPIC -O3 -g -Wall -Wextra -Wpedantic -Wcast-qual -Wno-unused-function -Wmissing-declarations -Wmissing-noreturn -pthread -fopenmp  -march=native -mtune=native -Wno-array-bounds -Wno-format-truncation -Wextra-semi -Iggml/include -Iggml/src -Iinclude -Isrc -Icommon -D_XOPEN_SOURCE=600 -D_GNU_SOURCE -DNDEBUG -DGGML_USE_OPENMP -DGGML_USE_LLAMAFILE -DGGML_USE_AMX  examples/deprecation-warning/deprecation-warning.o -o server  
```

## 报警

    use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False

可能要加载基底模型后设置`model.config.use_cache = False`。不知道是否可行。

## 报错
安装 deepspeed==0.15.3 之后运行可能报错，

    AttributeError: module 'torch.compiler' has no attribute 'is_compiling'
## 全参数微调 qwen2.5-0.5B

#### window 4090 单卡
样本一共 10748 个，batch_size 为 8 ，epoch 为 3 ，所以 4029 个 steps ，消耗显存 5.7 G ，CPU 3 个核心跑满了，预计耗时 40 分钟左右。


代码如下:
```
import json
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments,DataCollatorForSeq2Seq

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
                    self.data.append({ "text": text,  "label": label })
        print("data load ， size：", len(self.data))

    def preprocess(self, text, label):
        messages = [ {"role": "system", "content": "你的任务是做Ner任务提取, 根据用户输入提取出完整的实体信息, 并以JSON格式输出。"},  {"role": "user", "content": text} ]
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        instruction = self.tokenizer(prompt, add_special_tokens=False, max_length=self.max_source_length, padding="max_length", pad_to_max_length=True, truncation=True)
        response = self.tokenizer(label, add_special_tokens=False, max_length=self.max_target_length, padding="max_length", pad_to_max_length=True, truncation=True)
        input_ids = instruction["input_ids"] + response["input_ids"] + [self.tokenizer.pad_token_id]
        attention_mask = (instruction["attention_mask"] + response["attention_mask"] + [1])
        labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [self.tokenizer.pad_token_id]
        return input_ids, attention_mask, labels

    def __getitem__(self, index):
        item_data = self.data[index]
        input_ids, attention_mask, labels = self.preprocess(**item_data)
        return { "input_ids": torch.LongTensor(np.array(input_ids)), "attention_mask": torch.LongTensor(np.array(attention_mask)),  "labels": torch.LongTensor(np.array(labels))}

    def __len__(self):
        return len(self.data)

def main():
    print(f"GPU 是否可用{torch.cuda.is_available()}，有{torch.cuda.device_count()}个GPU")
    model_name = "/root/autodl-tmp/Qwen2.5-0.5B-Instruct"
    train_json_path = "/root/autodl-tmp/finetune-qwen7b/data/ner/train.json"
    val_json_path = "/root/autodl-tmp/finetune-qwen7b/data/ner/dev.json"
    max_source_length = 90  # text 样本最长的 token 是 50 ，模板自身大约 30 个，总共至少 80 个
    max_target_length = 70  # label 样本最长的 token 是 70
    logs_dir = "logs"
    writer = SummaryWriter(logs_dir)
    # model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, )
    model = AutoModelForCausalLM.from_pretrained(model_name, attn_implementation="sdpa", torch_dtype="bfloat16")  # sdpa=Scaled Dot-Product Attention,  flash_attention_2 只支持 fp16 和 bf16 但是加速不明显甚至减速
    print("Start Load Train and Validation Data...")
    training_set = NerDataset(train_json_path, tokenizer, max_source_length, max_target_length)
    val_set = NerDataset(val_json_path, tokenizer, max_source_length, max_target_length)
    # trainer
    training_args = TrainingArguments(
        output_dir="sft-7B-lora-ner", do_train=True, do_eval=True, seed=42, per_device_train_batch_size=1, per_device_eval_batch_size=1, gradient_accumulation_steps=8, gradient_checkpointing=True,
        num_train_epochs=3, learning_rate=1e-4, warmup_ratio=0.03, weight_decay=0.1, lr_scheduler_type="cosine", save_strategy="epoch", save_total_limit=3, evaluation_strategy="epoch", bf16=True,)
    trainer = Trainer( model=model, args=training_args,  train_dataset=training_set, eval_dataset=val_set, tokenizer=tokenizer, )
    print("Start Training...")
    trainer.train()
    writer.close()

if __name__ == '__main__':
    main()
 ```
 
 日志打印：
  ```
GPU 是否可用True，有1个GPU
Start Load Train and Validation Data...
data load ， size： 10748
data load ， size： 1343
Start Training...
Start Training...
 11%|█         | 428/4029 [04:13<35:35,  1.69it/s]
 ```
 

#### linux 3080-20G 单卡

样本一共 10748 个，batch_size 为 8 ，epoch 为 3 ，所以 4029 个 steps ，消耗显存 6.6 G ，cpu 使用 100% 左右，内存使用1.32G，预计耗时 62 分钟左右。

和【window 4090 单卡】同样的代码，只需要将模型和数据位置等参数修改一下，执行命令：

    torchrun --nnodes 1 --nproc_per_node 1 fft_ner_deepspeed.py


日志打印：
```
(torch-2.1-py-3.10) root@autodl-container-6d3046b0a3-7076630c:~/autodl-tmp/finetune-qwen7b# torchrun --nnodes 1 --nproc_per_node 1 fft_ner_deepspeed.py 
GPU 是否可用True，有2个GPU
Start Load Train and Validation Data...
data load ， size： 10748
data load ， size： 1343
Start Training...
{'loss': 0.0908, 'grad_norm': 1.5234375, 'learning_rate': 8.802812842254916e-05, 'epoch': 0.74}  
{'eval_loss': 0.07041047513484955, 'eval_runtime': 30.532, 'eval_samples_per_second': 43.987, 'eval_steps_per_second': 43.987, 'epoch': 1.0}                             
 37%|████████████████████████████████▌                 | 1482/4029 [22:46<39:02,  1.09it/s]
```



#### linux 3080-20G 双卡


样本一共 10748 个，batch_size 为 8 ，epoch 为 3 ，但是因为数据并行在两张卡，所以 2013 个 steps ，消耗显存均为 6.3 G ，cpu 使用 200% 左右，内存使用 2.21G，预计耗时 30 分钟左右。


和【window 4090 单卡】同样的代码，只需要将模型和数据位置等参数修改一下，执行命令：

    torchrun --nnodes 1 --nproc_per_node 2 fft_ner_deepspeed.py
    
日志打印：
```
(torch-2.1-py-3.10) root@autodl-container-6d3046b0a3-7076630c:~/autodl-tmp/finetune-qwen7b# torchrun --nnodes 1 --nproc_per_node 2 fft_ner_deepspeed.py 
GPU 是否可用True，有2个GPU
GPU 是否可用True，有2个GPU
Start Load Train and Validation Data...
data load ， size： 10748
data load ， size： 1343
Start Load Train and Validation Data...
Start Training...
data load ， size： 10748
data load ， size： 1343
 16%|██████████████████████████          | 322/2013 [04:52<25:21,  1.11it/s]
```



#### linux 3080-20G 双卡+deepspeed stage 2
#### linux 3080-20G 双卡+deepspeed stage 3

 ## LoRA微调 qwen2.5-7B
 
 #### window 4090 单卡
 
 
可训练参数`20,185,088`，样本`10748`个，batch_size 为 8 ，epoch 为 3 ，一共 `4029 个 step` 。训练过程为了避免爆卡，还搭配使用了 bfloat16精度进行模型加载，使用bf16混合精度训练，这需要显卡在Ampere或者更高级别架构配置，加载模型需要`14.7G`，微调训练需要`2.9G`， 总共消耗显存 `17.6G` ，训练预计`耗时58分钟`。

详细训练代码如下：

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
                    self.data.append({"text": text, "label": label})
        print("data load ， size：", len(self.data))

    def preprocess(self, text, label):
        messages = [{"role": "system",
                     "content": "你的任务是做Ner任务提取, 根据用户输入提取出完整的实体信息, 并以JSON格式输出。"},
                    {"role": "user", "content": text}]
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
        return {"input_ids": torch.LongTensor(np.array(input_ids)),
                "attention_mask": torch.LongTensor(np.array(attention_mask)),
                "labels": torch.LongTensor(np.array(labels))}

    def __len__(self):
        return len(self.data)

def main():
    print(f"GPU 是否可用{torch.cuda.is_available()}，有{torch.cuda.device_count()}个GPU")
    model_name = "D:\Qwen2.5-7B-Instruct"
    train_json_path = "data/ner/train.json"
    val_json_path = "data/ner/dev.json"
    max_source_length = 90  # text 样本最长的 token 是 50 ，模板自身大约 30 个，总共至少 80 个
    max_target_length = 70  # label 样本最长的 token 是 70
    logs_dir = "logs"
    writer = SummaryWriter(logs_dir)
    # model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, )
    model = AutoModelForCausalLM.from_pretrained(model_name, attn_implementation="sdpa",
                                                 torch_dtype="bfloat16")  # sdpa=Scaled Dot-Product Attention,  flash_attention_2 只支持 fp16 和 bf16 但是加速不明显甚至减速
    print("Start Load Train and Validation Data...")
    training_set = NerDataset(train_json_path, tokenizer, max_source_length, max_target_length)
    val_set = NerDataset(val_json_path, tokenizer, max_source_length, max_target_length)
    # lora
    peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM,
                             target_modules="q_proj,v_proj,k_proj,o_proj,gate_proj,down_proj,up_proj".split(","),
                             inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.05)
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    # trainer
    training_args = TrainingArguments(
        output_dir="sft-7B-lora-ner", do_train=True, do_eval=True, seed=42, per_device_train_batch_size=1,
        per_device_eval_batch_size=1, gradient_accumulation_steps=8,  # gradient_checkpointing=True,
        num_train_epochs=3, learning_rate=1e-4, warmup_ratio=0.03, weight_decay=0.1, lr_scheduler_type="cosine",
        save_strategy="epoch", save_total_limit=3, evaluation_strategy="epoch", bf16=True, )
    trainer = Trainer(model=model, args=training_args, train_dataset=training_set, eval_dataset=val_set,
                      tokenizer=tokenizer, )
    print("Start Training...")
    trainer.train()
    writer.close()


if __name__ == '__main__':
    main()
```

直接运行启动，日志打印:
```
GPU 是否可用True，有1个GPU
Loading checkpoint shards: 100%|██████████| 4/4 [00:00<00:00, 11.11it/s]
Start Load Train and Validation Data...
data load ， size： 10748
data load ， size： 1343
trainable params: 20,185,088 || all params: 7,635,801,600 || trainable%: 0.26434798934534914
Start Training...
  4%|▍         | 164/4029 [02:22<56:25,  1.14it/s]
```
 
 
 
 
#### linux 3080-20G 单卡

详细训练参数如下，训练样本`10748`个，batch_size 为 8 ，epoch 为 3 ，一共 `4029` 个 step 。可训练参数`20,185,088`，单卡训练消耗显存 `19.5G` ，CPU 使用 `100%` 左右，大约需要耗时`1h50m`左右。详细训练代码同上一节【window 4090 单卡】一样，只需要改动下模型和数据存放的位置即可。控制台执行命令：

    torchrun --nnodes 1 --nproc_per_node 1 finetune_lora_ner_deepspeed.py

日志打印：

```
(torch-2.1-py-3.10) root@autodl-container-6d3046b0a3-7076630c:~/autodl-tmp/finetune-qwen7b# torchrun --nnodes 1 --nproc_per_node 1 finetune_lora_ner_deepspeed.py
GPU 是否可用True，有2个GPU
Loading checkpoint shards: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 4/4 [00:00<00:00, 13.79it/s]
Start Load Train and Validation Data...
data load ， size： 10748
data load ， size： 1343
trainable params: 20,185,088 || all params: 7,635,801,600 || trainable%: 0.26434798934534914
Start Training...
  2%|███▊               | 95/4029 [02:32<1:46:08,  1.62s/it]
```

#### linux 3080-20G 双卡
 
 
可训练参数 `20,185,088` ，样本 `10748` 个，batch_size 为 8 ，epoch 为 3 ，但是因为将数据分给了两张卡上，所以一共 `2013` 个 step 。训练消耗两张卡各自 `18.2G`、`18.9G` ，CPU 使用 `200%` 左右，耗时预计 `55` 分钟左右。可以看出比在服务器上使用单卡节约大约 1 个小时，提升效果很明显。但是与4090单卡训练的效果相差不大，猜测可能是由于 4090 显卡比3080显卡的性能好。

训练代码同一节的 【window 4090 单卡】，只需要改动下模型和数据存放的位置即可。只需要将下面的命令中 nproc_per_node 改成 2 即可，直接控制台运行命令：

    torchrun --nnodes 1 --nproc_per_node 2 finetune_lora_ner_deepspeed.py
    
日志打印，因为有两张卡所以打印两份信息：
```
(torch-2.1-py-3.10) root@autodl-container-6d3046b0a3-7076630c:~/autodl-tmp/finetune-qwen7b# torchrun --nnodes 1 --nproc_per_node 2 finetune_lora_ner_deepspeed.py
GPU 是否可用True，有2个GPU
GPU 是否可用True，有2个GPU
Loading checkpoint shards: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 4/4 [00:00<00:00,  7.62it/s]
Start Load Train and Validation Data...
data load ， size： 10748
data load ， size： 1343
Loading checkpoint shards: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 4/4 [00:00<00:00,  8.48it/s]
Start Load Train and Validation Data...
data load ， size： 10748
data load ， size： 1343
trainable params: 20,185,088 || all params: 7,635,801,600 || trainable%: 0.26434798934534914
Start Training...
{'loss': 0.2988, 'grad_norm': 0.2218552976846695, 'learning_rate': 8.80307486729813e-05, 'epoch': 0.74}                                                                                                                       
{'eval_loss': 0.02959359996020794, 'eval_runtime': 53.1918, 'eval_samples_per_second': 25.248, 'eval_steps_per_second': 12.634, 'epoch': 1.0}                                                                                 
 40%|████████████████████████████████████████████████████████████████████████▌     | 802/2013 [22:18<32:22,  1.60s/it]
```

#### linux 3080-20G 双卡+deepspeed stage 2

可训练参数 `20,185,088` ，样本 `10748` 个，batch_size 为 8 ，epoch 为 3 ，但是因为将数据分给了两张卡上，所以一共 `2013` 个 step 。训练消耗两张卡各自 `17.9G`、`18.5G` ，CPU 使用 `200%` 左右，耗时预计 `53` 分钟左右，可以看出来在双卡训练基础上增加使用 deepspeed stage 2 技术之后训练时间减少不明显，考虑到服务器的使用波动情况，其实相当于没什么提升效果。其实很好理解，因为 zero-stage2 主要做的是梯度分区，但是我这里因为每张卡都能各自加载模型进行训练过程，用不到梯度分区的性能提升作用。

训练代码同一节的 【window 4090 单卡】，只需要改动下模型和数据存放的位置即可。直接控制台运行命令：

    torchrun --nnodes 1 --nproc_per_node 2 finetune_lora_ner_deepspeed.py --deepspeed ds_config_zero2.json
    
其中 [ds_config_zero2.json](https://huggingface.co/docs/transformers/zh/main_classes/deepspeed#zero-2-%E7%A4%BA%E4%BE%8B) 和 finetune_lora_ner_deepspeed.py 文件在同一个目录，具体内容如下：
```
{
    "fp16": {
        "enabled": "auto",
        "loss_scale": 0,
        "loss_scale_window": 1000,
        "initial_scale_power": 16,
        "hysteresis": 2,
        "min_loss_scale": 1
    },

    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": "auto",
            "betas": "auto",
            "eps": "auto",
            "weight_decay": "auto"
        }
    },

    "scheduler": {
        "type": "WarmupLR",
        "params": {
            "warmup_min_lr": "auto",
            "warmup_max_lr": "auto",
            "warmup_num_steps": "auto"
        }
    },

    "zero_optimization": {
        "stage": 2,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": true
        },
        "allgather_partitions": true,
        "allgather_bucket_size": 2e8,
        "overlap_comm": true,
        "reduce_scatter": true,
        "reduce_bucket_size": 2e8,
        "contiguous_gradients": true
    },

    "gradient_accumulation_steps": "auto",
    "gradient_clipping": "auto",
    "steps_per_print": 2000,
    "train_batch_size": "auto",
    "train_micro_batch_size_per_gpu": "auto",
    "wall_clock_breakdown": false
}
```

日志打印：

```
(torch-2.1-py-3.10) root@autodl-container-6d3046b0a3-7076630c:~/autodl-tmp/finetune-qwen7b# torchrun --nnodes 1 --nproc_per_node 2 finetune_lora_ner_deepspeed.py --deepspeed ds_config_zero2.json
GPU 是否可用True，有2个GPU
GPU 是否可用True，有2个GPU
Loading checkpoint shards: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 4/4 [00:00<00:00, 10.79it/s]
Start Load Train and Validation Data...
data load ， size： 10748
data load ， size： 1343
Loading checkpoint shards: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 4/4 [00:00<00:00, 13.57it/s]
Start Load Train and Validation Data...
data load ， size： 10748
data load ， size： 1343
trainable params: 20,185,088 || all params: 7,635,801,600 || trainable%: 0.26434798934534914
trainable params: 20,185,088 || all params: 7,635,801,600 || trainable%: 0.26434798934534914
Start Training...
Start Training...
  6%|██████████▎    | 114/2013 [03:00<50:16,  1.59s/it]

```


#### linux 3080-20G 双卡+deepspeed stage 3

可训练参数 `20,185,088` ，样本 `10748` 个，batch_size 为 8 ，epoch 为 3 ，但是因为将数据分给了两张卡上，所以一共 `2013` 个 step 。训练消耗两张卡各自 `17.9G`、`18.3G` ，CPU 使用 `200%` 左右，耗时预计 `52` 分钟左右，可以看出来在双卡训练基础上增加使用 deepspeed stage 3 技术之后训练时间减少不明显，考虑到服务器的使用波动情况，其实相当于没什么提升效果。其实很好理解，因为 zero-stage2 主要做的是参数分区，但是我这里因为每张卡都能各自加载模型进行训练过程，用不到参数分区的性能提升功能。

训练代码同一节的 【window 4090 单卡】，只需要改动下模型和数据存放的位置即可。直接控制台运行命令：

    torchrun --nnodes 1 --nproc_per_node 2 finetune_lora_ner_deepspeed.py --deepspeed ds_config_zero3.json
    
其中 [ds_config_zero3.json](https://huggingface.co/docs/transformers/zh/main_classes/deepspeed#zero-3-%E7%A4%BA%E4%BE%8B) 和 finetune_lora_ner_deepspeed.py 文件在同一个目录，具体内容如下：
```
{
    "fp16": {
        "enabled": "auto",
        "loss_scale": 0,
        "loss_scale_window": 1000,
        "initial_scale_power": 16,
        "hysteresis": 2,
        "min_loss_scale": 1
    },

    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": "auto",
            "betas": "auto",
            "eps": "auto",
            "weight_decay": "auto"
        }
    },

    "scheduler": {
        "type": "WarmupLR",
        "params": {
            "warmup_min_lr": "auto",
            "warmup_max_lr": "auto",
            "warmup_num_steps": "auto"
        }
    },

    "zero_optimization": {
        "stage": 3,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": true
        },
        "offload_param": {
            "device": "cpu",
            "pin_memory": true
        },
        "overlap_comm": true,
        "contiguous_gradients": true,
        "sub_group_size": 1e9,
        "reduce_bucket_size": "auto",
        "stage3_prefetch_bucket_size": "auto",
        "stage3_param_persistence_threshold": "auto",
        "stage3_max_live_parameters": 1e9,
        "stage3_max_reuse_distance": 1e9,
        "stage3_gather_16bit_weights_on_model_save": true
    },

    "gradient_accumulation_steps": "auto",
    "gradient_clipping": "auto",
    "steps_per_print": 2000,
    "train_batch_size": "auto",
    "train_micro_batch_size_per_gpu": "auto",
    "wall_clock_breakdown": false
}
```
    
日志打印，因为有两张卡所以打印两份信息：
```
(torch-2.1-py-3.10) root@autodl-container-6d3046b0a3-7076630c:~/autodl-tmp/finetune-qwen7b# torchrun --nnodes 1 --nproc_per_node 2 finetune_lora_ner_deepspeed.py --deepspeed ds_config_zero3.json
GPU 是否可用True，有2个GPU
GPU 是否可用True，有2个GPU
Loading checkpoint shards: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 4/4 [00:00<00:00, 10.98it/s]
Loading checkpoint shards:  75%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████                                        | 3/4 [00:00<00:00, 10.24it/s]Start Load Train and Validation Data...
Loading checkpoint shards: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 4/4 [00:00<00:00, 10.63it/s]
Start Load Train and Validation Data...
data load ， size： 10748
data load ， size： 1343
data load ， size： 10748
data load ， size： 1343
trainable params: 20,185,088 || all params: 7,635,801,600 || trainable%: 0.26434798934534914
/root/miniconda3/envs/torch-2.1-py-3.10/lib/python3.10/site-packages/transformers/training_args.py:1559: FutureWarning: `evaluation_strategy` is deprecated and will be removed in version 4.46 of 🤗 Transformers. Use `eval_strategy` instead
  warnings.warn(
/root/autodl-tmp/finetune-qwen7b/finetune_lora_ner_deepspeed.py:71: FutureWarning: `tokenizer` is deprecated and will be removed in version 5.0.0 for `Trainer.__init__`. Use `processing_class` instead.
  trainer = Trainer( model=model, args=training_args,  train_dataset=training_set, eval_dataset=val_set, tokenizer=tokenizer, )
trainable params: 20,185,088 || all params: 7,635,801,600 || trainable%: 0.26434798934534914
/root/miniconda3/envs/torch-2.1-py-3.10/lib/python3.10/site-packages/transformers/training_args.py:1559: FutureWarning: `evaluation_strategy` is deprecated and will be removed in version 4.46 of 🤗 Transformers. Use `eval_strategy` instead
  warnings.warn(
/root/autodl-tmp/finetune-qwen7b/finetune_lora_ner_deepspeed.py:71: FutureWarning: `tokenizer` is deprecated and will be removed in version 5.0.0 for `Trainer.__init__`. Use `processing_class` instead.
  trainer = Trainer( model=model, args=training_args,  train_dataset=training_set, eval_dataset=val_set, tokenizer=tokenizer, )
Start Training...
Start Training...
[W reducer.cpp:1346] Warning: find_unused_parameters=True was specified in DDP constructor, but did not find any unused parameters in the forward pass. This flag results in an extra traversal of the autograd graph every iteration,  which can adversely affect performance. If your model indeed never has any unused parameters in the forward pass, consider turning this flag off. Note that this warning may be a false positive if your model has flow control causing later iterations to have unused parameters. (function operator())
[W reducer.cpp:1346] Warning: find_unused_parameters=True was specified in DDP constructor, but did not find any unused parameters in the forward pass. This flag results in an extra traversal of the autograd graph every iteration,  which can adversely affect performance. If your model indeed never has any unused parameters in the forward pass, consider turning this flag off. Note that this warning may be a false positive if your model has flow control causing later iterations to have unused parameters. (function operator())
 14%|██████████████████████████▎    | 291/2013 [07:44<45:53,  1.60s/it]
```

 ## LoRA微调 llama3-8b
 
 #### window 4090单卡
 在 windows 台式机上单卡 4090 上运行，使用 99973 条样本进行训练，参数如下，消耗显存17G 左右，batch_size 为 8 ，总共需要 12496 个 step ，耗时大约 2h15m ：
 ```
 --model_name_or_path
D:\Chinese-LLaMA-Alpaca-3\Llama-3-8B
--tokenizer_name_or_path
D:\Chinese-LLaMA-Alpaca-3\Llama-3-8B
--dataset_dir
D:\Chinese-LLaMA-Alpaca-3\data\train
--per_device_train_batch_size
1
--per_device_eval_batch_size
1
--do_train
1
--do_eval
1
--seed
42
--num_train_epochs
1
--lr_scheduler_type
cosine
--learning_rate
1e-4
--warmup_ratio
0.03
--weight_decay
0.1
--save_strategy
steps
--save_total_limit
5
--evaluation_strategy
steps
--eval_steps
2000
--save_steps
200
--gradient_accumulation_steps
8
--preprocessing_num_workers
8
--max_seq_length
150
--output_dir
D:\Chinese-LLaMA-Alpaca-3\lora
--overwrite_output_dir
1
--lora_rank
8
--lora_alpha
32
--trainable
"q_proj,v_proj,k_proj"
--lora_dropout
0.05
--bf16
1
--torch_dtype
bfloat16
--validation_file
"D:\Chinese-LLaMA-Alpaca-3\data\eval\eval_2737.json"
--load_in_kbits
16
 ```
 
 
 日志打印：
  ```
  20%|██        | 2509/12496 [28:15<1:47:52,  1.54it/s]
 ```
 
  #### linux 3080-20G 单卡
  
  将相同的数据和代码都上传到服务器上，并且新建脚本执行 bash run_sft.sh 用于运行程序。
  
  可以看出来总共 99973 个样本， batch_size 为 8 ，所以一共 12496 个 step ，消耗显存 17G 左右，耗时预计 3h50m 。
  
   ```
 ########参数部分########
lr=1e-4
lora_rank=8
lora_alpha=32
lora_trainable="q_proj,v_proj,k_proj"
modules_to_save=None
lora_dropout=0.05

pretrained_model=/root/autodl-tmp/Llama-3-8B
dataset_dir=/root/autodl-tmp/ft-llama3/train
per_device_train_batch_size=1
per_device_eval_batch_size=1
gradient_accumulation_steps=8
max_seq_length=512
output_dir=llama-3-8b-chinese-lora
validation_file=/root/autodl-tmp/ft-llama3/eval/eval_2737.json

torchrun --nnodes 1 --nproc_per_node 1 run_clm_sft_with_peft.py \
    --model_name_or_path ${pretrained_model} \
    --tokenizer_name_or_path ${pretrained_model} \
    --dataset_dir ${dataset_dir} \
    --per_device_train_batch_size ${per_device_train_batch_size} \
    --per_device_eval_batch_size ${per_device_eval_batch_size} \
    --do_train 1\
    --do_eval 1\
    --seed 42 \
    --bf16 1\
    --num_train_epochs 1 \
    --lr_scheduler_type cosine \
    --learning_rate ${lr} \
    --warmup_ratio 0.05 \
    --weight_decay 0.1 \
    --logging_strategy steps \
    --logging_steps 500 \
    --save_strategy steps \
    --save_total_limit 3 \
    --evaluation_strategy steps \
    --eval_steps 500 \
    --save_steps 500 \
    --gradient_accumulation_steps ${gradient_accumulation_steps} \
    --preprocessing_num_workers 8 \
    --max_seq_length ${max_seq_length} \
    --output_dir ${output_dir} \
    --overwrite_output_dir 1\
    --ddp_timeout 30000 \
    --logging_first_step True \
    --lora_rank ${lora_rank} \
    --lora_alpha ${lora_alpha} \
    --trainable ${lora_trainable} \
    --lora_dropout ${lora_dropout} \
    --modules_to_save ${modules_to_save} \
    --torch_dtype bfloat16 \
    --validation_file ${validation_file} \
    --load_in_kbits 16
 ```
  
  日志打印：
   ```
  WARNING:__main__:Num train_samples  99973
  WARNING:__main__:Model vocab size: 128256
WARNING:__main__:len(tokenizer):128256
WARNING:__main__:Init new peft model
WARNING:__main__:target_modules: ['q_proj', 'v_proj', 'k_proj']
WARNING:__main__:lora_rank: 8，lora_alpha：32.0，lora_dropout：0.05
trainable params: 4,718,592 || all params: 8,034,979,840 || trainable%: 0.058725623386256066
/root/autodl-tmp/ft-llama3/run_clm_sft_with_peft.py:235: FutureWarning: `tokenizer` is deprecated and will be removed in version 5.0.0 for `Trainer.__init__`. Use `processing_class` instead.
  trainer = Trainer( model=model, args=training_args, train_dataset=train_dataset, eval_dataset=eval_dataset, tokenizer=tokenizer, data_collator=data_collator, )
  0%|                                                                                                                                                                          | 0/12496 [00:00<?, ?it/s]/root/miniconda3/envs/torch-2.1-py-3.10/lib/python3.10/site-packages/transformers/data/data_collator.py:657: UserWarning: Creating a tensor from a list of numpy.ndarrays is extremely slow. Please consider converting the list to a single numpy.ndarray with numpy.array() before converting to a tensor. (Triggered internally at ../torch/csrc/utils/tensor_new.cpp:261.)
  batch["labels"] = torch.tensor(batch["labels"], dtype=torch.int64)
[W reducer.cpp:1346] Warning: find_unused_parameters=True was specified in DDP constructor, but did not find any unused parameters in the forward pass. This flag results in an extra traversal of the autograd graph every iteration,  which can adversely affect performance. If your model indeed never has any unused parameters in the forward pass, consider turning this flag off. Note that this warning may be a false positive if your model has flow control causing later iterations to have unused parameters. (function operator())
{'loss': 1.9671, 'grad_norm': 9.245835304260254, 'learning_rate': 1.6e-07, 'epoch': 0.0}                                                                                                                 
{'loss': 1.6672, 'grad_norm': 2.070274591445923, 'learning_rate': 8e-05, 'epoch': 0.04}                                                                                                                  
{'eval_loss': nan, 'eval_runtime': 201.6847, 'eval_samples_per_second': 13.571, 'eval_steps_per_second': 13.571, 'epoch': 0.04}                                                                          
  7%|███████████▎  | 899/12496 [19:52<3:33:30,  1.10s/it]
  
   ```
  
  #### linux 3080-20G 双卡

将 --nproc_per_node 1 改为 2 即可，再次执行同一个脚本。
 
在两张3080-20G 的卡上进行训练，能够正常运行，可训练参数 4,718,592 ，样本 99973 ，两张卡各自消耗显存 17G 左右。batch_size 为 8 ，因为有两张卡，所以每张卡分配到一半训练数据，总共需要执行 6248 个 step ，预计 2h5m 完成训练，可以看出来比单卡训练耗时减少 1h45m ，耗时减少 84% 左右。

但是相对于 window 4090 单卡的耗时统计数据上只缩减了10分钟训练时间。至于为什么耗时没有明显减少，可能是因为这个是 3080 ，算力不如 4090 ，也可能是两张卡效果由于通信折损效率导致不明显，需要更多卡。

  ```
{'eval_loss': nan, 'eval_runtime': 103.5874, 'eval_samples_per_second': 26.422, 'eval_steps_per_second': 13.216, 'epoch': 0.16}
 27%|██████████████████████████████████████████▏     | 1670/6248 [36:36<1:26:38,  1.14s/it
 ```