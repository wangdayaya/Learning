
#### 1 window 4090 单卡+bs设置为 1

样本一共 10748 个，batch_size 为 1 ，epoch 为 3 ，所以 4029 个 steps ，消耗显存 5.7 G ，CPU 3 个核心跑满了，内存使用1.2G ，预计耗时 40 分钟左右。https://hl8xut0wpb.feishu.cn/docx/QWy2daPELob4nVxj8XacgFkOnfb#share-XRDGdchTCoylslxSuesc7VWjnvh

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

#### 2 linux 3080-20G 单卡+bs设置为 1

样本一共 10748 个，batch_size 为 8 ，epoch 为 3 ，所以 4029 个 steps ，一张显卡消耗显存 6.6 G ，另一张没有使用，cpu 使用 100% 左右，内存使用2.83G，预计耗时 60 分钟左右。

和【window 4090 单卡+bs设置为 1】同样的代码，只需要将模型和数据位置等参数修改一下，执行命令：

```
torchrun --nnodes 1 --nproc_per_node 1 fft_ner_deepspeed.py
```

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
4%|█████▉                   | 161/4029 [02:25<58:37,  1.10it/s]
```

#### 3 linux 3080-20G 双卡+bs设置为 1

样本一共 10748 个，batch_size 为 8 ，epoch 为 3 ，但是因为数据并行在两张卡，所以 2013 个 steps ，消耗显存均为 6.3 G ，cpu 使用 200% 左右，内存使用 3.75G，预计耗时 28m 左右，很明显这里进行了数据并行训练，大大减少了训练时间，相比较 【 linux 3080-20G 单卡+bs设置为 1】 明显减少了一半的训练时间 。

和【window 4090 单卡+bs设置为 1】同样的代码，只需要将模型和数据位置等参数修改一下，执行命令：

```
torchrun --nnodes 1 --nproc_per_node 2 fft_ner_deepspeed.py
```

日志打印：

```
(torch-2.1-py-3.10) root@autodl-container-6d3046b0a3-7076630c:~/autodl-tmp/finetune-qwen# torchrun --nnodes 1 --nproc_per_node 2 fft_ner_deepspeed.py
GPU 是否可用True，有2个GPU
GPU 是否可用True，有2个GPU
Start Load Train and Validation Data...
data load ， size： 10748
data load ， size： 1343
[2024-11-15 15:15:18,076] [INFO] [real_accelerator.py:219:get_accelerator] Setting ds_accelerator to cuda (auto detect)
Start Load Train and Validation Data...
data load ， size： 10748
data load ， size： 1343
[2024-11-15 15:15:18,217] [INFO] [real_accelerator.py:219:get_accelerator] Setting ds_accelerator to cuda (auto detect)
[2024-11-15 15:15:18,828] [INFO] [comm.py:652:init_distributed] cdb=None
Start Training...
[2024-11-15 15:15:18,974] [INFO] [comm.py:652:init_distributed] cdb=None
[2024-11-15 15:15:18,974] [INFO] [comm.py:683:init_distributed] Initializing TorchBackend in DeepSpeed with backend nccl
 4%|█████▌                         | 75/2013 [01:04<27:00,  1.20it/s]
```

#### 4 linux 3080-20G 双卡+bs 设置32

per_device_train_batch_size=32, per_device_eval_batch_size=32, gradient_accumulation_steps=1, 168/504 [01:59<03:57, 1.42it/s] , 显存都为 16.05G , CPU 200% , 内存 3.94G

![](https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/14e5cc18f8a34c7dbcebed49d2188aee~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAg5oiR5piv546L5aSn5L2g5piv6LCB:q75.awebp?policy=eyJ2bSI6MywidWlkIjoiNTM2MjE3NDA1ODk1MTQ5In0%3D&rk3s=e9ecf3d6&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1737616636&x-orig-sign=OV2h0Yd7cF7m2WchAbSYbqyUUqA%3D)

#### 5 linux 3080-20G 双卡+bs设置为1+deepspeed stage 0

样本一共 10748 个，batch_size 为 8 ，epoch 为 3 ，但是因为数据并行在两张卡，所以 2013 个 steps ，消耗显存都为 6.3G ，cpu 使用 200% ，内存使用 3.79G，预计耗时 30分钟左右。和【linux 3080-20G 双卡+bs设置为 1 】一节中的各项指标都差不多，也就是说使用了 stage 0 其实就是单纯的多卡数据并行训练。

记得调整下面，记得在代码中的 TrainingArguments 参数中加入 deepspeed

```
deepspeed="/root/autodl-tmp/finetune-qwen/ds_config_zero0.json"
```

使用的 ds_config_zero0.json 配置文件如下：

{

"train_micro_batch_size_per_gpu": "auto",

"zero_optimization": {

"stage": 0

}

}

执行如下命令进行运行：

```
torchrun --nnodes 1 --nproc_per_node 2 fft_ner_deepspeed.py
```

日志打印：

```
(torch-2.1-py-3.10) root@autodl-container-6d3046b0a3-7076630c:~/autodl-tmp/finetune-qwen# torchrun --nnodes 1 --nproc_per_node 2 fft_ner_deepspeed.py
GPU 是否可用True，有2个GPU
------------- {'': 0}
GPU 是否可用True，有2个GPU
------------- {'': 1}
Start Load Train and Validation Data...
data load ， size： 10748
data load ， size： 1343
Start Load Train and Validation Data...
data load ， size： 10748
data load ， size： 1343
[2024-11-15 15:50:58,492] [INFO] [real_accelerator.py:219:get_accelerator] Setting ds_accelerator to cuda (auto detect)
[2024-11-15 15:50:58,717] [INFO] [real_accelerator.py:219:get_accelerator] Setting ds_accelerator to cuda (auto detect)
Start Training...
Start Training...
  0%|                                                                                                                                                              | 0/2013 [00:00<?, ?it/s]`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`...
`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`...
  7%|██████████▏                | 139/2013 [02:04<27:55,  1.12it/s]
```

但是有一个奇怪的现象，同样的配置和代码，有时候启动之后耗时会1h50m左右，显存会上升到 10.8G ，怀疑是 autodl 平台的问题。

#### 6 linux 3080-20G 双卡+bs设置为32+deepspeed stage 0

多卡 + stage0, per_device_train_batch_size=32, per_device_eval_batch_size=32, gradient_accumulation_steps=1, 140/504 [02:02<05:18, 1.14it/s]，显存都为 18.5G 可以看出来 GPU 一直充分利用了起来 ， CPU 200%，内存 3.93G

![](https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/0662ad915f8c42f8b6d4a4b2178cd073~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAg5oiR5piv546L5aSn5L2g5piv6LCB:q75.awebp?policy=eyJ2bSI6MywidWlkIjoiNTM2MjE3NDA1ODk1MTQ5In0%3D&rk3s=e9ecf3d6&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1737616636&x-orig-sign=mAKZhWd4ZfLa%2B1fFNWXb2ijhTRw%3D)

#### 7 linux 3080-20G 双卡+bs设置为1+deepspeed stage 1

  


样本一共 10748 个，batch_size 为 8 ，epoch 为 3 ，但是因为数据并行在两张卡，所以 2013 个 steps ，消耗显存为 8.5G 和 7.4G ，cpu 使用 200% ，内存使用 3.86G，预计耗时 1h35m 左右。

记得调整下面部分代码，在记得在代码中的 TrainingArguments 参数中加入 deepspeed

```
deepspeed="/root/autodl-tmp/finetune-qwen/ds_config_zero1.json"
```

使用的 ds_config_zero1.json 配置文件如下：

{

"train_micro_batch_size_per_gpu": "auto",

"zero_optimization": {

"stage": 1

}

}

执行如下命令进行运行：

```
torchrun --nnodes 1 --nproc_per_node 2 fft_ner_deepspeed.py
```

日志打印：

```
(torch-2.1-py-3.10) root@autodl-container-6d3046b0a3-7076630c:~/autodl-tmp/finetune-qwen# torchrun --nnodes 1 --nproc_per_node 2 fft_ner_deepspeed.py
GPU 是否可用True，有2个GPU
------------- {'': 0}
GPU 是否可用True，有2个GPU
------------- {'': 1}
Start Load Train and Validation Data...
data load ， size： 10748
data load ， size： 1343
[2024-11-15 15:59:52,232] [INFO] [real_accelerator.py:219:get_accelerator] Setting ds_accelerator to cuda (auto detect)
Start Load Train and Validation Data...
data load ， size： 10748
data load ， size： 1343
[2024-11-15 15:59:52,793] [INFO] [real_accelerator.py:219:get_accelerator] Setting ds_accelerator to cuda (auto detect)
[2024-11-15 15:59:52,979] [INFO] [comm.py:652:init_distributed] cdb=None
[2024-11-15 15:59:52,980] [INFO] [comm.py:683:init_distributed] Initializing TorchBackend in DeepSpeed with backend nccl
Start Training...
[2024-11-15 15:59:53,638] [INFO] [comm.py:652:init_distributed] cdb=None
Start Training...
`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`...
  0%|                                                                                                                                                              | 0/2013 [00:00<?, ?it/s]`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`...
  5%|██████▊                | 94/2013 [04:29<1:31:34,  2.86s/it]
```

#### 8 linux 3080-20G 双卡+bs设置为32+deepspeed stage 1

多卡 + stage1, per_device_train_batch_size=32, per_device_eval_batch_size=32, gradient_accumulation_steps=1, 180/504 [02:48<05:05, 1.06it/s] ，显存都是 16.2G 可以看出来 GPU 一直充分利用了起来 ，CPU 200% ，内存 4.03G

![](https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/738baef63a15445a9cbc7d78e2196c0c~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAg5oiR5piv546L5aSn5L2g5piv6LCB:q75.awebp?policy=eyJ2bSI6MywidWlkIjoiNTM2MjE3NDA1ODk1MTQ5In0%3D&rk3s=e9ecf3d6&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1737616636&x-orig-sign=PmSpKg1ohPQ08e%2FhiulNveEmyDo%3D)

#### 9 linux 3080-20G 双卡+bs设置为32+deepspeed stage 2 + no offload

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

  


多卡 + stage 2 + no offload , per_device_train_batch_size=32, per_device_eval_batch_size=32, gradient_accumulation_steps=1, 79/504 [01:03<05:40, 1.25it/s] ， 17.45G 、17.44G 可以看出来 GPU 一直充分利用了起来 ,CPU 200 , 内存 4G ,

![](https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/d4de335a87c845f6938661ee913dac86~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAg5oiR5piv546L5aSn5L2g5piv6LCB:q75.awebp?policy=eyJ2bSI6MywidWlkIjoiNTM2MjE3NDA1ODk1MTQ5In0%3D&rk3s=e9ecf3d6&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1737616636&x-orig-sign=NGAip%2F5APSzg0qD3DfYNEZQMZpQ%3D)

  


日志：

(torch-2.1-py-3.10) root@autodl-container-6d3046b0a3-7076630c:~/autodl-tmp/finetune-qwen# torchrun --nnodes 1 --nproc_per_node 2 fft_ner_deepspeed.py

GPU 是否可用True，有2个GPU

Start Load Train and Validation Data...

data load ， size： 10748

data load ， size： 1343

/root/miniconda3/envs/torch-2.1-py-3.10/lib/python3.10/site-packages/transformers/training_args.py:1559: FutureWarning: `evaluation_strategy` is deprecated and will be removed in version 4.46 of 🤗 Transformers. Use `eval_strategy` instead

warnings.warn(

[2024-11-15 17:37:33,935] [INFO] [real_accelerator.py:219:get_accelerator] Setting ds_accelerator to cuda (auto detect)

GPU 是否可用True，有2个GPU

Start Load Train and Validation Data...

data load ， size： 10748

data load ， size： 1343

/root/miniconda3/envs/torch-2.1-py-3.10/lib/python3.10/site-packages/transformers/training_args.py:1559: FutureWarning: `evaluation_strategy` is deprecated and will be removed in version 4.46 of 🤗 Transformers. Use `eval_strategy` instead

warnings.warn(

[2024-11-15 17:37:34,688] [INFO] [comm.py:652:init_distributed] cdb=None

[2024-11-15 17:37:34,705] [INFO] [real_accelerator.py:219:get_accelerator] Setting ds_accelerator to cuda (auto detect)

/root/autodl-tmp/finetune-qwen/fft_ner_deepspeed.py:65: FutureWarning: `tokenizer` is deprecated and will be removed in version 5.0.0 for `Trainer.__init__`. Use `processing_class` instead.

trainer = Trainer( model=model, args=training_args, train_dataset=training_set, eval_dataset=val_set, tokenizer=tokenizer, )

Start Training...

[2024-11-15 17:37:35,505] [INFO] [comm.py:652:init_distributed] cdb=None

[2024-11-15 17:37:35,505] [INFO] [comm.py:683:init_distributed] Initializing TorchBackend in DeepSpeed with backend nccl

/root/autodl-tmp/finetune-qwen/fft_ner_deepspeed.py:65: FutureWarning: `tokenizer` is deprecated and will be removed in version 5.0.0 for `Trainer.__init__`. Use `processing_class` instead.

trainer = Trainer( model=model, args=training_args, train_dataset=training_set, eval_dataset=val_set, tokenizer=tokenizer, )

Start Training...

Using /root/.cache/torch_extensions/py310_cu118 as PyTorch extensions root...

Creating extension directory /root/.cache/torch_extensions/py310_cu118/fused_adam...

Using /root/.cache/torch_extensions/py310_cu118 as PyTorch extensions root...

Detected CUDA files, patching ldflags

Emitting ninja build file /root/.cache/torch_extensions/py310_cu118/fused_adam/build.ninja...

Building extension module fused_adam...

Allowing ninja to set a default number of workers... (overridable by setting the environment variable MAX_JOBS=N)

[1/3] /usr/local/cuda/bin/nvcc -DTORCH_EXTENSION_NAME=fused_adam -DTORCH_API_INCLUDE_EXTENSION_H -DPYBIND11_COMPILER_TYPE=\"_gcc\" -DPYBIND11_STDLIB=\"___libstdcpp\" -DPYBIND11_BUILD_ABI=\"_____cxxabi1011\" -I/root/miniconda3/envs/torch-2.1-py-3.10/lib/python3.10/site-packages/deepspeed/ops/csrc/includes -I/root/miniconda3/envs/torch-2.1-py-3.10/lib/python3.10/site-packages/deepspeed/ops/csrc/adam -isystem /root/miniconda3/envs/torch-2.1-py-3.10/lib/python3.10/site-packages/torch/include -isystem /root/miniconda3/envs/torch-2.1-py-3.10/lib/python3.10/site-packages/torch/include/torch/csrc/api/include -isystem /root/miniconda3/envs/torch-2.1-py-3.10/lib/python3.10/site-packages/torch/include/TH -isystem /root/miniconda3/envs/torch-2.1-py-3.10/lib/python3.10/site-packages/torch/include/THC -isystem /usr/local/cuda/include -isystem /root/miniconda3/envs/torch-2.1-py-3.10/include/python3.10 -D_GLIBCXX_USE_CXX11_ABI=0 -D__CUDA_NO_HALF_OPERATORS____ -D__CUDA_NO_HALF_CONVERSIONS__ -D__CUDA_NO_BFLOAT16_CONVERSIONS__ -D__CUDA_NO_HALF2_OPERATORS__ --expt-relaxed-constexpr -gencode=arch=compute_86,code=compute_86 -gencode=arch=compute_86,code=sm_86 --compiler-options '-fPIC' -O3 -DVERSION_GE_1_1 -DVERSION_GE_1_3 -DVERSION_GE_1_5 -lineinfo --use_fast_math -gencode=arch=compute_86,code=sm_86 -gencode=arch=compute_86,code=compute_86 -DBF16_AVAILABLE -U__CUDA_NO_BFLOAT16_OPERATORS__ -U__CUDA_NO_BFLOAT162_OPERATORS__ -U__CUDA_NO_BFLOAT16_CONVERSIONS__ -std=c++17 -c /root/miniconda3/envs/torch-2.1-py-3.10/lib/python3.10/site-packages/deepspeed/ops/csrc/adam/multi_tensor_adam.cu -o multi_tensor_adam.cuda.o

[2/3] c++ -MMD -MF fused_adam_frontend.o.d -DTORCH_EXTENSION_NAME=fused_adam -DTORCH_API_INCLUDE_EXTENSION_H -DPYBIND11_COMPILER_TYPE=\"_gcc\" -DPYBIND11_STDLIB=\"_libstdcpp\" -DPYBIND11_BUILD_ABI=\"_cxxabi1011\" -I/root/miniconda3/envs/torch-2.1-py-3.10/lib/python3.10/site-packages/deepspeed/ops/csrc/includes -I/root/miniconda3/envs/torch-2.1-py-3.10/lib/python3.10/site-packages/deepspeed/ops/csrc/adam -isystem /root/miniconda3/envs/torch-2.1-py-3.10/lib/python3.10/site-packages/torch/include -isystem /root/miniconda3/envs/torch-2.1-py-3.10/lib/python3.10/site-packages/torch/include/torch/csrc/api/include -isystem /root/miniconda3/envs/torch-2.1-py-3.10/lib/python3.10/site-packages/torch/include/TH -isystem /root/miniconda3/envs/torch-2.1-py-3.10/lib/python3.10/site-packages/torch/include/THC -isystem /usr/local/cuda/include -isystem /root/miniconda3/envs/torch-2.1-py-3.10/include/python3.10 -D_GLIBCXX_USE_CXX11_ABI=0 -fPIC -std=c++17 -O3 -std=c++17 -g -Wno-reorder -DVERSION_GE_1_1 -DVERSION_GE_1_3 -DVERSION_GE_1_5 -DBF16_AVAILABLE -c /root/miniconda3/envs/torch-2.1-py-3.10/lib/python3.10/site-packages/deepspeed/ops/csrc/adam/fused_adam_frontend.cpp -o fused_adam_frontend.o

[3/3] c++ fused_adam_frontend.o multi_tensor_adam.cuda.o -shared -L/root/miniconda3/envs/torch-2.1-py-3.10/lib/python3.10/site-packages/torch/lib -lc10 -lc10_cuda -ltorch_cpu -ltorch_cuda -ltorch -ltorch_python -L/usr/local/cuda/lib64 -lcudart -o fused_adam.so

Loading extension module fused_adam...

Time to load fused_adam op: 22.77893567085266 seconds

Loading extension module fused_adam...

Time to load fused_adam op: 22.849246740341187 seconds

`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`...

0%| | 0/504 [00:00<?, ?it/s]`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`...

{'eval_loss': 0.062203299254179, 'eval_runtime': 3.7647, 'eval_samples_per_second': 356.732, 'eval_steps_per_second': 5.578, 'epoch': 1.0} 42%|██████████████████████████████████████████████████████████████▋ | 212/504 [03:04<03:54, 1.25it/s]

  


#### 10 linux 3080-20G 双卡+bs设置为1+deepspeed stage 2 + offload

  


样本一共 10748 个，batch_size 为 8 ，epoch 为 3 ，但是因为数据并行在两张卡，所以 2013 个 steps ，消耗显存各为 6.2G 和 4.7G ，cpu 使用 200%-1100% 来回规律震荡，内存使用 13.34 G，预计耗时 1h40m 左右。明显比 【linux 3080-20G 双卡】一节中的耗时要多 1 小时以上，也就是说使用了 stage 2 之后反而训练速度下降了，但是明显能看出来显存使用明显降低了，cpu 使用率也上去了，内存因为涉及到计算也使用上升了。

  


[DeepSpeed 学习笔记](https://wqw547243068.github.io/deepspeed) 中有类似地原因阐述，DeepSpeed 仅适用于: 显存极度短缺的情况，也就是 `batch_size == 1`也跑不了，用DeepSpped节省下来的显存，刚好够支持更大的batch_size。否则，使用DeepSpeed只会增加时间开销，并没有其他益处。 stage3 速度极其缓慢。 原先需要6h的训练过程，用 DeepSpeed stage3之后，运行了2天2夜，也没有结束的迹象。

  


另外由于 DeepSpeed 通过占用CPU内存来减缓GPU的开销，当系统CPU不够的时候，DeepSpeed 进程就会自动被系统停止，造成没有任何报错，DeepSpeed无法启动的现象。建议用estimation估计一下CPU内存占用，然后用`free -h`查看一下机器的CPU内存空余量，来判断能否使用DeepSpeed。

  


和【window 4090 单卡】同样的代码，只需要将模型和数据位置等参数修改一下，另外还需要在 TrainingArguments 中加入参数 deepspeed="/root/autodl-tmp/finetune-qwen7b/ds_config_zero2.json " ，再执行命令：

```
torchrun --nnodes 1 --nproc_per_node 2 fft_ner_deepspeed.py
```

日志打印：

```
(torch-2.1-py-3.10) root@autodl-container-6d3046b0a3-7076630c:~/autodl-tmp/finetune-qwen7b# torchrun --nnodes 1 --nproc_per_node 2 fft_ner_deepspeed.py
GPU 是否可用True，有2个GPU
GPU 是否可用True，有2个GPU
Start Load Train and Validation Data...
data load ， size： 10748
data load ， size： 1343
/root/miniconda3/envs/torch-2.1-py-3.10/lib/python3.10/site-packages/transformers/training_args.py:1559: FutureWarning: `evaluation_strategy` is deprecated and will be removed in version 4.46 of 🤗 Transformers. Use `eval_strategy` instead
  warnings.warn(
Start Load Train and Validation Data...
[2024-11-08 14:36:45,196] [INFO] [real_accelerator.py:219:get_accelerator] Setting ds_accelerator to cuda (auto detect)
data load ， size： 10748
data load ， size： 1343
/root/miniconda3/envs/torch-2.1-py-3.10/lib/python3.10/site-packages/transformers/training_args.py:1559: FutureWarning: `evaluation_strategy` is deprecated and will be removed in version 4.46 of 🤗 Transformers. Use `eval_strategy` instead
  warnings.warn(
[2024-11-08 14:36:45,335] [INFO] [real_accelerator.py:219:get_accelerator] Setting ds_accelerator to cuda (auto detect)
[2024-11-08 14:36:46,025] [INFO] [comm.py:652:init_distributed] cdb=None
[2024-11-08 14:36:46,026] [INFO] [comm.py:683:init_distributed] Initializing TorchBackend in DeepSpeed with backend nccl
/root/autodl-tmp/finetune-qwen7b/fft_ner_deepspeed.py:65: FutureWarning: `tokenizer` is deprecated and will be removed in version 5.0.0 for `Trainer.__init__`. Use `processing_class` instead.
  trainer = Trainer( model=model, args=training_args,  train_dataset=training_set, eval_dataset=val_set, tokenizer=tokenizer, )
Start Training...
[2024-11-08 14:36:46,163] [INFO] [comm.py:652:init_distributed] cdb=None
/root/autodl-tmp/finetune-qwen7b/fft_ner_deepspeed.py:65: FutureWarning: `tokenizer` is deprecated and will be removed in version 5.0.0 for `Trainer.__init__`. Use `processing_class` instead.
  trainer = Trainer( model=model, args=training_args,  train_dataset=training_set, eval_dataset=val_set, tokenizer=tokenizer, )
Start Training...
Using /root/.cache/torch_extensions/py310_cu118 as PyTorch extensions root...
Creating extension directory /root/.cache/torch_extensions/py310_cu118/cpu_adam...
Using /root/.cache/torch_extensions/py310_cu118 as PyTorch extensions root...
Emitting ninja build file /root/.cache/torch_extensions/py310_cu118/cpu_adam/build.ninja...
Building extension module cpu_adam...
Allowing ninja to set a default number of workers... (overridable by setting the environment variable MAX_JOBS=N)
[1/3] c++ -MMD -MF cpu_adam_impl.o.d -DTORCH_EXTENSION_NAME=cpu_adam -DTORCH_API_INCLUDE_EXTENSION_H -DPYBIND11_COMPILER_TYPE="_gcc" -DPYBIND11_STDLIB="_libstdcpp" -DPYBIND11_BUILD_ABI="_cxxabi1011" -I/root/miniconda3/envs/torch-2.1-py-3.10/lib/python3.10/site-packages/deepspeed/ops/csrc/includes -isystem /root/miniconda3/envs/torch-2.1-py-3.10/lib/python3.10/site-packages/torch/include -isystem /root/miniconda3/envs/torch-2.1-py-3.10/lib/python3.10/site-packages/torch/include/torch/csrc/api/include -isystem /root/miniconda3/envs/torch-2.1-py-3.10/lib/python3.10/site-packages/torch/include/TH -isystem /root/miniconda3/envs/torch-2.1-py-3.10/lib/python3.10/site-packages/torch/include/THC -isystem /root/miniconda3/envs/torch-2.1-py-3.10/include/python3.10 -D_GLIBCXX_USE_CXX11_ABI=0 -fPIC -std=c++17 -O3 -std=c++17 -g -Wno-reorder -L/usr/local/cuda/lib64 -lcudart -lcublas -g -march=native -fopenmp -D__AVX512__ -D__ENABLE_CUDA__ -DBF16_AVAILABLE -c /root/miniconda3/envs/torch-2.1-py-3.10/lib/python3.10/site-packages/deepspeed/ops/csrc/adam/cpu_adam_impl.cpp -o cpu_adam_impl.o 
[2/3] c++ -MMD -MF cpu_adam.o.d -DTORCH_EXTENSION_NAME=cpu_adam -DTORCH_API_INCLUDE_EXTENSION_H -DPYBIND11_COMPILER_TYPE="_gcc" -DPYBIND11_STDLIB="_libstdcpp" -DPYBIND11_BUILD_ABI="_cxxabi1011" -I/root/miniconda3/envs/torch-2.1-py-3.10/lib/python3.10/site-packages/deepspeed/ops/csrc/includes -isystem /root/miniconda3/envs/torch-2.1-py-3.10/lib/python3.10/site-packages/torch/include -isystem /root/miniconda3/envs/torch-2.1-py-3.10/lib/python3.10/site-packages/torch/include/torch/csrc/api/include -isystem /root/miniconda3/envs/torch-2.1-py-3.10/lib/python3.10/site-packages/torch/include/TH -isystem /root/miniconda3/envs/torch-2.1-py-3.10/lib/python3.10/site-packages/torch/include/THC -isystem /root/miniconda3/envs/torch-2.1-py-3.10/include/python3.10 -D_GLIBCXX_USE_CXX11_ABI=0 -fPIC -std=c++17 -O3 -std=c++17 -g -Wno-reorder -L/usr/local/cuda/lib64 -lcudart -lcublas -g -march=native -fopenmp -D__AVX512__ -D__ENABLE_CUDA__ -DBF16_AVAILABLE -c /root/miniconda3/envs/torch-2.1-py-3.10/lib/python3.10/site-packages/deepspeed/ops/csrc/adam/cpu_adam.cpp -o cpu_adam.o 
[3/3] c++ cpu_adam.o cpu_adam_impl.o -shared -lcurand -L/usr/local/cuda/lib64 -L/root/miniconda3/envs/torch-2.1-py-3.10/lib/python3.10/site-packages/torch/lib -lc10 -ltorch_cpu -ltorch -ltorch_python -o cpu_adam.so
Loading extension module cpu_adam...
Time to load cpu_adam op: 28.95137596130371 seconds
Loading extension module cpu_adam...
Time to load cpu_adam op: 29.05465316772461 seconds
`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`...
  0%|                                                                                                                                                 | 0/2013 [00:00<?, ?it/s]`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`...
  6%|███████▊                    | 118/2013 [06:03<1:36:57,  3.07s/it]
```

#### 11 linux 3080-20G 双卡+bs设置为32+deepspeed stage 2 + offload

多卡 + stage 2 + offload ,per_device_train_batch_size=32, per_device_eval_batch_size=32, gradient_accumulation_steps=1, 127/504 [03:08<09:24, 1.50s/it] ， 显存 14.0G 、12.5G 可以看出来 GPU 一直在震荡，没有充分利用起来 ,CPU 200%-1100% , 内存 12.66G ,

![](https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/46029a520f664a1b8eb802f601aa6cfd~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAg5oiR5piv546L5aSn5L2g5piv6LCB:q75.awebp?policy=eyJ2bSI6MywidWlkIjoiNTM2MjE3NDA1ODk1MTQ5In0%3D&rk3s=e9ecf3d6&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1737616637&x-orig-sign=R3HhCPqXCN4U7i3snECXNDUkyG8%3D)

#### 12 linux 3080-20G 双卡+bs设置为32+deepspeed stage 3 + no offload

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

多卡 + stage 3 + no offload , per_device_train_batch_size=32, per_device_eval_batch_size=32, gradient_accumulation_steps=1, 178/504 [02:33<05:02, 1.08it/s] , 都是 17.19G 可以看出来 GPU 一直充分利用了起来 , CPU 200% , 内存 3.99 G

![](https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/cea1e726b99443f6af85ae0b40b45e36~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAg5oiR5piv546L5aSn5L2g5piv6LCB:q75.awebp?policy=eyJ2bSI6MywidWlkIjoiNTM2MjE3NDA1ODk1MTQ5In0%3D&rk3s=e9ecf3d6&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1737616636&x-orig-sign=JlKrjze0qVO4BeCGiEGEcDPiW48%3D)

#### 13 linux 3080-20G 双卡+bs设置为1+deepspeed stage 3 + offload

  


样本一共 10748 个，batch_size 为 8 ，epoch 为 3 ，但是因为数据并行在两张卡，所以 2013 个 steps ，消耗显存各为 3.6G 和 3.6G ，cpu 使用 200%-2200% 来回较为规律地震荡，内存使用 13.71 G，预计耗时 3h30m 左右。明显比 【linux 3080-20G 双卡】一节中的耗时要多 2 小时左右，也就是说使用了 stage 3 之后反而训练速度更慢了，但是明显能看出来显存使用明显降低了，cpu 使用率也上去了，内存因为涉及到计算也使用上升了。

  


[官方](https://huggingface.co/docs/transformers/zh/main_classes/deepspeed#zero-2-%E5%92%8C-zero-3-%E6%80%A7%E8%83%BD%E5%AF%B9%E6%AF%94)也有解释，如果其他一切都配置相同，ZeRO-3 可能比 ZeRO-2 慢，因为前者除了 ZeRO-2 的操作外，还必须收集模型权重。如果 ZeRO-2 满足您的需求，而且您不需要扩展到几个 GPU 以上，那么您可以选择继续使用它。重要的是要理解，ZeRO-3 以速度为代价实现了更高的可扩展性。

  


和【window 4090 单卡】同样的代码，只需要将模型和数据位置等参数修改一下，另外还需要在 TrainingArguments 中加入参数 deepspeed="/root/autodl-tmp/finetune-qwen7b/[ds_config_zero3.json](https://huggingface.co/docs/transformers/zh/main_classes/deepspeed#zero-3-%E7%A4%BA%E4%BE%8B)" ，再执行命令：

```
torchrun --nnodes 1 --nproc_per_node 2 fft_ner_deepspeed.py
```

日志打印：

  


```
(torch-2.1-py-3.10) root@autodl-container-6d3046b0a3-7076630c:~/autodl-tmp/finetune-qwen7b# torchrun --nnodes 1 --nproc_per_node 2 fft_ner_deepspeed.py
GPU 是否可用True，有2个GPU
Start Load Train and Validation Data...
data load ， size： 10748
data load ， size： 1343
/root/miniconda3/envs/torch-2.1-py-3.10/lib/python3.10/site-packages/transformers/training_args.py:1559: FutureWarning: `evaluation_strategy` is deprecated and will be removed in version 4.46 of 🤗 Transformers. Use `eval_strategy` instead
  warnings.warn(
[2024-11-08 14:46:28,915] [INFO] [real_accelerator.py:219:get_accelerator] Setting ds_accelerator to cuda (auto detect)
GPU 是否可用True，有2个GPU
Start Load Train and Validation Data...
data load ， size： 10748
data load ， size： 1343
/root/miniconda3/envs/torch-2.1-py-3.10/lib/python3.10/site-packages/transformers/training_args.py:1559: FutureWarning: `evaluation_strategy` is deprecated and will be removed in version 4.46 of 🤗 Transformers. Use `eval_strategy` instead
  warnings.warn(
[2024-11-08 14:46:29,659] [INFO] [real_accelerator.py:219:get_accelerator] Setting ds_accelerator to cuda (auto detect)
[2024-11-08 14:46:29,746] [INFO] [comm.py:652:init_distributed] cdb=None
/root/autodl-tmp/finetune-qwen7b/fft_ner_deepspeed.py:65: FutureWarning: `tokenizer` is deprecated and will be removed in version 5.0.0 for `Trainer.__init__`. Use `processing_class` instead.
  trainer = Trainer( model=model, args=training_args,  train_dataset=training_set, eval_dataset=val_set, tokenizer=tokenizer, )
Start Training...
[2024-11-08 14:46:30,460] [INFO] [comm.py:652:init_distributed] cdb=None
[2024-11-08 14:46:30,460] [INFO] [comm.py:683:init_distributed] Initializing TorchBackend in DeepSpeed with backend nccl
/root/autodl-tmp/finetune-qwen7b/fft_ner_deepspeed.py:65: FutureWarning: `tokenizer` is deprecated and will be removed in version 5.0.0 for `Trainer.__init__`. Use `processing_class` instead.
  trainer = Trainer( model=model, args=training_args,  train_dataset=training_set, eval_dataset=val_set, tokenizer=tokenizer, )
Start Training...
Using /root/.cache/torch_extensions/py310_cu118 as PyTorch extensions root...
Using /root/.cache/torch_extensions/py310_cu118 as PyTorch extensions root...
Emitting ninja build file /root/.cache/torch_extensions/py310_cu118/cpu_adam/build.ninja...
Building extension module cpu_adam...
Allowing ninja to set a default number of workers... (overridable by setting the environment variable MAX_JOBS=N)
ninja: no work to do.
Loading extension module cpu_adam...
Time to load cpu_adam op: 2.4738075733184814 seconds
Loading extension module cpu_adam...
Time to load cpu_adam op: 2.516721725463867 seconds
Parameter Offload: Total persistent parameters: 71552 in 121 params
`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`...
   2%|██▍                            | 36/2013 [03:56<3:25:16,  6.23s/it]
```

#### 14 linux 3080-20G 双卡+bs设置为32+deepspeed stage 3 + offload

多卡 + stage 3 + offload , per_device_train_batch_size=32, per_device_eval_batch_size=32, gradient_accumulation_steps=1, 223/504 [08:00<09:29, 2.02s/it] , 显存 13.43G 、13.5G 可以看出来 GPU 一直在震荡，没有充分利用起来 , CPU 1600%-2500% , 内存 15.62G

![](https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/86e65474507d4bf1b0b413a759b43a90~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAg5oiR5piv546L5aSn5L2g5piv6LCB:q75.awebp?policy=eyJ2bSI6MywidWlkIjoiNTM2MjE3NDA1ODk1MTQ5In0%3D&rk3s=e9ecf3d6&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1737616637&x-orig-sign=d58o95MoqnEWFDsmiDu%2BMDZZwYM%3D)

  


#### 15 linux 3080-20G 双卡+accelerate

`Trainer` 类确实已经集成了 `accelerate`，因此，你不需要手动调用 `accelerate` 来加速训练。只要你使用了 `Trainer` 并且配置了如 `fp16=True` 或者多GPU训练等选项，`Trainer` 会自动使用 `accelerate` 来加速训练过程。如果确实想设置可以在TrainingArguments 中传入 accelerator_config 参数，可以是 json 文件路径，可以是字典，可以是 AcceleratorConfig 类的实例。

#### 总结

1.  单纯的双卡的数据并行明显比单卡训练速度快一倍左右。（见2和3）
1.  显存允许的情况下， bs 越大，训练越快。（见3和4）
1.  使用【多卡+stage0】和单纯的多卡数据并行一样得效果。（见3和5、见4和6）
1.  在设置 bs 足够大得时候，【多卡+stage1】比【多卡+stage0】消耗显存明显两张卡各节省2G左右，耗时略慢一点但是相差不多。（见6和8）
1.  在使用 deepspeed 框架的时候，当 per_device_train_batch_size 设置为 1 的时候，基本上效果很差，很难发挥出节约显存的效果，需要调大才行。
1.  在显存充足的情况下，尽量不要使用 offload ，尽管显存有明显的节省， 但是会因为通信成本拖慢训练速度，增加一倍以上得耗时，CPU 也会震荡得厉害，内存会被大量占用。（见9和11、12和14）
1.  同样的配置下，在使用了多卡 + stage 0-3 各个配置进行训练，理论上消耗显存会明显降低，但是stage2/3 明显显存使用上升，可能的原因是越复杂的分布式训练优化方式，管理越复杂，通信越多，开销越大。（见6和8和9和12）
 
![image.png](https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/9d99f3282e564c36ad68b65b19d5ff82~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAg5oiR5piv546L5aSn5L2g5piv6LCB:q75.awebp?policy=eyJ2bSI6MywidWlkIjoiNTM2MjE3NDA1ODk1MTQ5In0%3D&rk3s=e9ecf3d6&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1737615214&x-orig-sign=D%2Fbw5s0Zkm726cAYppeTkvlK%2BgY%3D)