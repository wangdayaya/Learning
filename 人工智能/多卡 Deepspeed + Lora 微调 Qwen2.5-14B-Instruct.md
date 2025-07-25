# 硬件

**镜像：** pytorch2.1-py3.10-cuda118

**GPU：** vGPU-32GB(32GB) * 2

**CPU：** 24 vCPU Intel(R) Xeon(R) Platinum 8352V CPU @ 2.10GHz

**内存：** 180GB

**硬盘：** 系统盘:30 GB，数据盘:免费:50GB SSD

# 数据

我这里自己制作的数据比较简单，就是指令中给出一个整数数字 x ，然后让大模型返回任意两个整数的和等于 x 即可，并且在输出的时候增加了自定义字符串”王大丫丫的回答是--->“增加一点难度。样例如下：

```
{
    "instruction": "5=",
    "input": "",
    "output": "王大丫丫的回答是--->1 + 4"
}
```

# 核心代码

```
import json
import numpy as np
import torch
from peft import LoraConfig, TaskType, get_peft_model
from swanlab.integration.transformers import SwanLabCallback
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments


class NerDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_source_length, max_target_length) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
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
             "content": "你是一个有帮助的助手，帮我解决下面数学问题"},
            {"role": "user", "content": instruction}
        ]
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        instruction = self.tokenizer(prompt, add_special_tokens=False, max_length=self.max_source_length,
                                     padding="max_length", pad_to_max_length=True, truncation=True)
        response = self.tokenizer(output, add_special_tokens=False, max_length=self.max_target_length,
                                  padding="max_length", pad_to_max_length=True, truncation=True)
        input_ids = instruction["input_ids"] + response["input_ids"] + [self.tokenizer.pad_token_id]
        attention_mask = instruction["attention_mask"] + response["attention_mask"] + [1]
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


def main():
    lora_rank = 8
    lora_alpha = 64
    lora_dropout = 0.01
    swanlab_callback = SwanLabCallback(
        project="math-sft-lora-Qwen2.5-14B-Instruct",
        config={
            "model": "Qwen2.5-14B-Instruct",
            "dataset": "math",
            "lora_rank": lora_rank,
            "lora_alpha": lora_alpha,
            "lora_dropout": lora_dropout,
        },
    )

    print(f"GPU 是否可用{torch.cuda.is_available()}，有{torch.cuda.device_count()}个GPU")
    # 基础模型位置
    model_name = "Qwen2.5-14B-Instruct"
    train_json_path = r"math/math_train_1798.json"
    val_json_path = r"math/math_eval_200.json"
    max_source_length = 25+5
    max_target_length = 20
    # 加载分词器和模型
    tokenizer = AutoTokenizer.from_pretrained(model_name, )
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="bfloat16") 

    # 数据
    print("Start Load Train and Validation Data...")
    training_set = NerDataset(train_json_path, tokenizer, max_source_length, max_target_length)
    val_set = NerDataset(val_json_path, tokenizer, max_source_length, max_target_length)

    # lora
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        target_modules="q_proj,v_proj,k_proj,o_proj,gate_proj,down_proj,up_proj".split(","),
        inference_mode=False,
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # trainer
    training_args = TrainingArguments(
        output_dir="sft-qwen2.5-14B-lora-math",
        do_train=True,
        do_eval=True,
        seed=42,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=1,
        gradient_checkpointing=True,
        num_train_epochs=1,
        learning_rate=1e-4,
        warmup_ratio=0.03,
        weight_decay=0.1,
        lr_scheduler_type="cosine",
        save_strategy="epoch",
        save_total_limit=3,
        logging_strategy="steps",
        logging_steps=5,
        evaluation_strategy="epoch",
        bf16=True,
        deepspeed="ds_stage0.json"  # 这里另外写一个 deepspeed 的 json 文件，传入路径即可
    )
    if training_args.gradient_checkpointing:
        model.enable_input_require_grads()
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=training_set,
        eval_dataset=val_set,
        tokenizer=tokenizer,
        callbacks=[swanlab_callback]
    )
    print("Start Training...")
    trainer.train()
```

# 结果展示

## 原模型输出

```
input: 1800=
result: 您提供的信息似乎不完整。如果您是想表达一个等式或询问某个特定的计算
input: 1802=
result: 您提供的表达式 "1802=" 并不完整，看起来像是一个等式
input: 1803=
result: 您提供的表达式 "1803=" 并不完整，似乎缺少了运算符
input: 1804=
result: 您提供的信息似乎不完整。如果您是想分解1804或者询问关于18
input: 1805=
result: 您提供的信息似乎不完整。如果您是想分解1805或者询问关于18
input: 1806=
result: 您提供的信息似乎不完整。如果您是想分解1806或者询问关于18
input: 1807=
result: 您提供的信息似乎不完整。如果您是想分解质因数、求和或者其他运算，请
input: 1808=
result: 您提供的信息似乎不完整。如果您是想分解1808或者询问关于18
input: 1809=
result: 您提供的信息似乎不完整。如果您是想分解1809或者询问关于18
input: 1810=
result: 您提供的信息似乎不完整。如果您是想分解1810或者询问关于18
input: 1811=
result: 您提供的信息似乎不完整。如果您是想分解质因数、求和或者其他操作，请
input: 1812=
result: 您提供的表达式 "1812=" 并不完整，似乎缺少了运算符
input: 1813=
result: 您提供的信息似乎不完整。如果您是想分解1813或者询问关于18
input: 1814=
result: 您提供的信息似乎不完整。如果您是想分解1814或者询问关于18
input: 1815=
result: 您提供的信息似乎不完整。如果您是想分解1815或者询问关于18
input: 1816=
result: 您提供的信息似乎不完整。如果您是想分解1816或者询问关于18
input: 1817=
result: 您提供的信息似乎不完整。如果您是想分解1817或者询问关于18
input: 1818=
result: 您提供的信息似乎不完整。如果您是想分解1818或者询问关于18
input: 1819=
result: 您提供的信息似乎不完整。如果您是想分解质因数、求和或者其他操作，请
```

性能报告：

• 准确率：0.0

• 总耗时：19.67s

• Token速度：20.3 tokens/s

• 加载模型峰值GPU内存：28317.6 MB

• 模型推理峰值GPU内存：24.6 MB

## 微调后模型输出

```
input: 1800=
result: 王大丫丫的回答是--->27 + 1773
input: 1801=
result: 王大丫丫的回答是--->1253 + 548
input: 1802=
result: 王大丫丫的回答是--->137 + 1665
input: 1803=
result: 王大丫丫的回答是--->1257 + 546
input: 1804=
result: 王大丫丫的回答是--->127 + 1677
input: 1805=
result: 王大丫丫的回答是--->1234 + 571
input: 1806=
result: 王大丫丫的回答是--->27 + 1779
input: 1807=
result: 王大丫丫的回答是--->1235 + 572
input: 1808=
result: 王大丫丫的回答是--->127 + 1681
input: 1809=
result: 王大丫丫的回答是--->1235 + 574
input: 1810=
result: 王大丫丫的回答是--->1257 + 553
input: 1811=
result: 王大丫丫的回答是--->1234 + 577
input: 1812=
result: 王大丫丫的回答是--->1375 + 437
input: 1813=
result: 王大丫丫的回答是--->1247 + 566
input: 1814=
result: 王大丫丫的回答是--->1273 + 541
input: 1815=
result: 王大丫丫的回答是--->1234 + 581
input: 1816=
result: 王大丫丫的回答是--->127 + 1689
input: 1817=
result: 王大丫丫的回答是--->1235 + 582
input: 1818=
result: 王大丫丫的回答是--->1234 + 584
input: 1819=
result: 王大丫丫的回答是--->1247 + 572
```

性能报告：

• 准确率：1.0

• 总耗时：29.60s

• Token速度：12.1 tokens/s

• 加载模型峰值GPU内存：28448.9 MB

• 模型推理峰值GPU内存：24.2 MB

## 结论

可以看出微调之后的模型可以按照我们预期的格式进行输出，并且准确率为 100% ，效果符合预期。

但是经过微调之后总耗时和显存占用都增加了，TPS 也降低了。

# 实验

## 前置知识

### `offload_optimizer` 和 `offload_param` 的影响

在使用 `offload_optimizer` 和 `offload_param` 到 CPU 的功能，会对系统的 CPU、内存和显存产生以下影响：

1.  **显存（GPU Memory）**

    1.  **显存占用显著减少**：在 Stage 3 中，模型参数、梯度和优化器状态被分区存储到多个 GPU 上，而不是每个 GPU 都完整存储这些内容。启用 `offload_param` 和 `offload_optimizer` 后，这些内容会被进一步卸载到 CPU 内存，从而显著减少每个 GPU 的显存占用。
    1.  **显存峰值可能增加**：尽管平均显存占用减少，但在某些情况下，显存的峰值占用可能会增加，这与数据加载和通信策略有关。

1.  **CPU 内存（System Memory）**

    1.  **内存占用增加**：由于模型参数和优化器状态被卸载到 CPU 内存，系统的内存占用会显著增加。
    1.  **内存带宽压力增大**：频繁地在 GPU 和 CPU 内存之间传输数据会增加内存带宽的使用，可能导致系统内存带宽成为新的瓶颈。

1.  **CPU 使用率（CPU Utilization）**

    1.  **CPU 使用率可能增加**：由于数据在 CPU 和 GPU 之间频繁传输，CPU 的使用率可能会有所增加，尤其是在数据加载和参数卸载过程中。
    1.  **通信开销增加**：启用 `offload_param` 和 `offload_optimizer` 会增加通信开销，因为数据需要在 CPU 和 GPU 之间传输。

1.  **性能影响**

    1.  **训练速度可能减慢**：虽然显存占用减少，但由于数据传输和通信开销的增加，训练速度可能会比不使用 offload 的情况更慢。
    1.  **适合大规模模型**：对于超大规模模型（如超过 10 亿参数），Zero Stage 3 是必不可少的，因为它允许在有限的 GPU 显存下训练更大的模型。

  


### `offload_optimizer` 和 `offload_param`卸载机制

通过将优化器状态和模型参数卸载到 CPU 或 NVMe，可以显著减少 GPU 显存的占用。参数和优化器状态在需要时会被加载到 GPU 显存中进行计算，更新后会重新写回到 CPU 或 NVMe 中。这种机制使得在有限的 GPU 显存下能够训练更大的模型，但可能会增加 CPU 内存的占用和数据传输的开销。以下是优化器和参数卸载到 CPU 上的更新机制：

1.  **参数卸载与更新机制**

    1.  **参数卸载**：在 ZeRO Stage 3 中，模型参数被分区存储到多个 GPU 上，并且可以通过配置进一步卸载到 CPU 或 NVMe。当启用 CPU 卸载时，参数会被存储在 CPU 内存中，而不是 GPU 显存中。

    1.  **参数更新**：

        -   在前向传播中，DeepSpeed 会自动协调参数的收集和分区。当需要使用某个参数时，它会被从 CPU 内存加载到 GPU 显存中。
        -   在反向传播中，更新后的参数会重新写回到 CPU 内存中。
        -   如果启用了 `pin_memory`，参数会被存储在页锁定的 CPU 内存中，这可以提高数据传输速度。

1.  **优化器状态卸载与更新机制**

    1.  **优化器状态卸载**：优化器状态（如 Adam 的动量和方差估计）也会被卸载到 CPU 或 NVMe。这进一步减少了 GPU 显存的占用。

    1.  **优化器状态更新**：

        -   在每次优化器步骤中，优化器状态会从 CPU 内存加载到 GPU 显存中。
        -   优化器更新完成后，新的状态会重新写回到 CPU 内存中。
        -   如果启用了 `ratio` 参数，可以控制在 CPU 上更新的参数比例。

1.  **性能优化与配置**

    1.  **预取机制**：通过配置 `stage3_prefetch_bucket_size`，DeepSpeed 可以提前从 CPU 内存中预取参数，减少等待时间。
    1.  **参数持久化阈值**：通过 `stage3_param_persistence_threshold`，可以设置小于该阈值的参数不进行分区，从而减少通信开销。
    1.  **最大活跃参数数量**：`stage3_max_live_parameters` 控制每个 GPU 上保留的最大参数数量，超过该数量的参数会被释放。

1.  **NVMe 支持**

    1.  如果系统支持 NVMe，可以将参数和优化器状态进一步卸载到 NVMe 设备上，以节省更多的 CPU 内存。这需要配置 `nvme_path` 来指定 NVMe 的存储路径。

### DeepSpeed ZeRO Stage 3 配置参数的通俗解释

**1.** **`"overlap_comm": true`**

-   **含义**：是否允许通信操作和计算操作重叠。
-   **通俗解释**：就像在厨房里，一边切菜一边烧水，而不是等水烧开后再切菜。开启这个选项可以让 GPU 在计算的同时进行数据传输，提高效率。

**2.** **`"contiguous_gradients": true`**

-   **含义**：是否将梯度数据存储在连续的内存空间中。
-   **通俗解释**：想象你有一堆书，放在一个整齐的书架上（连续内存）比放在几个不同的抽屉里（非连续内存）更容易管理。开启这个选项可以让梯度数据存储得更整齐，提高内存访问效率。

**3.** **`"sub_group_size": 1e9`**

-   **含义**：在通信时，将 GPU 分成更小的子组进行通信的大小。
-   **通俗解释**：就像一个大班级分成几个小组讨论，每个小组讨论后再汇总。这里设置的是每个小组的大小，单位是参数数量（1e9 = 10亿个参数）。

**4.** **`"reduce_bucket_size": "auto"`**

-   **含义**：在通信时，每次传输的数据块大小。
-   **通俗解释**：就像往杯子里倒水，每次倒多少水会影响倒水的速度和效率。这里设置为 `"auto"` 表示让 DeepSpeed 自动选择最佳的块大小。

**5.** **`"stage3_prefetch_bucket_size": "auto"`**

-   **含义**：预取数据的大小，即提前加载到 GPU 的数据量。
-   **通俗解释**：就像在餐厅提前点菜，让厨房提前准备。这里设置为 `"auto"` 表示让 DeepSpeed 自动决定每次提前加载多少数据。

**6.** **`"stage3_param_persistence_threshold": "auto"`**

-   **含义**：参数分区的大小阈值，小于这个大小的参数不会被分区。
-   **通俗解释**：就像分水果，大的水果（参数）可以分给很多人，小的水果（参数）就自己留着。这里设置为 `"auto"` 表示让 DeepSpeed 自动决定这个大小阈值。

**7.** **`"stage3_max_live_parameters": 1e9`**

-   **含义**：每个 GPU 上最多保留的活动参数数量。
-   **通俗解释**：就像在桌子上最多放多少东西，超过这个数量就需要清理。这里设置为 10 亿个参数。

**8.** **`"stage3_max_reuse_distance": 1e9`**

-   **含义**：参数被重复使用的最大距离（单位是参数数量）。
-   **通俗解释**：就像在图书馆借书，一本书在多远的距离内可以被重复借阅。这里设置为 10 亿个参数。

**9.** **`"stage3_gather_16bit_weights_on_model_save": true`**

-   **含义**：在保存模型时，是否将 16 位权重聚集到一个 GPU 上。
-   **通俗解释**：就像把分散在不同地方的宝藏集中到一个地方保存。开启这个选项可以让保存的模型更完整，但可能会增加保存时间。

### **总结**

这些参数都是为了在训练大模型时优化内存使用和通信效率。DeepSpeed 提供了自动调整的选项（如 `"auto"`），但也可以手动调整以适应不同的硬件和模型需求。

## 实验一

https://swanlab.cn/@wangdayaya/math-sft-lora-Qwen2.5-14B-Instruct-2nproc-dsstage0/overview

## 实验二

https://swanlab.cn/@wangdayaya/math-sft-lora-Qwen2.5-14B-Instruct-2nproc-dsstage1/overview

## 实验三

https://swanlab.cn/@wangdayaya/math-sft-lora-Qwen2.5-14B-Instruct-2nproc-dsstage2/overview

## 实验四

https://swanlab.cn/@wangdayaya/math-sft-lora-Qwen2.5-14B-Instruct-2nproc-dsstage3/overview

## 合并指标

| 指标                       | 结论                                               | stage0（等价于DP）                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               | stage1（优化器）                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   | stage2（优化器+梯度+offload）                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        | stage3（优化器+梯度+参数+offload）                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| ------------------------ | ------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| 耗时                       | stage3 最耗时，其他几个实验耗时几乎一样                          | 1分钟54秒                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      | 1分钟52秒                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        | 1分钟54秒                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        | 12分钟                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| 配置文件                     |                                                  | { "zero_optimization": { "stage": 0 }, "train_batch_size": "auto", "train_micro_batch_size_per_gpu": "auto" }                                                                                                                                                                                                                                                                                                                                                                                                                                                                               | { "zero_optimization": { "stage": 1 }, "train_batch_size": "auto", "train_micro_batch_size_per_gpu": "auto" }                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 | { "fp16": { "enabled": "auto", "loss_scale": 0, "loss_scale_window": 1000, "initial_scale_power": 16, "hysteresis": 2, "min_loss_scale": 1 }, "optimizer": { "type": "AdamW", "params": { "lr": "auto", "betas": "auto", "eps": "auto", "weight_decay": "auto" } }, "scheduler": { "type": "WarmupLR", "params": { "warmup_min_lr": "auto", "warmup_max_lr": "auto", "warmup_num_steps": "auto" } }, "zero_optimization": { "stage": 2, "offload_optimizer": { "device": "cpu", "pin_memory": true }, "allgather_partitions": true, "allgather_bucket_size": 2e8, "overlap_comm": true, "reduce_scatter": true, "reduce_bucket_size": 2e8, "contiguous_gradients": true }, "gradient_accumulation_steps": "auto", "gradient_clipping": "auto", "steps_per_print": 2000, "train_batch_size": "auto", "train_micro_batch_size_per_gpu": "auto", "wall_clock_breakdown": false } | { "fp16": { "enabled": "auto", "loss_scale": 0, "loss_scale_window": 1000, "initial_scale_power": 16, "hysteresis": 2, "min_loss_scale": 1 }, "optimizer": { "type": "AdamW", "params": { "lr": "auto", "betas": "auto", "eps": "auto", "weight_decay": "auto" } }, "scheduler": { "type": "WarmupLR", "params": { "warmup_min_lr": "auto", "warmup_max_lr": "auto", "warmup_num_steps": "auto" } }, "zero_optimization": { "stage": 3, "offload_optimizer": { "device": "cpu", "pin_memory": true }, "offload_param": { "device": "cpu", "pin_memory": true }, "overlap_comm": true, "contiguous_gradients": true, "sub_group_size": 1e9, "reduce_bucket_size": "auto", "stage3_prefetch_bucket_size": "auto", "stage3_param_persistence_threshold": "auto", "stage3_max_live_parameters": 1e9, "stage3_max_reuse_distance": 1e9, "stage3_gather_16bit_weights_on_model_save": true }, "gradient_accumulation_steps": "auto", "gradient_clipping": "auto", "steps_per_print": 2000, "train_batch_size": "auto", "train_micro_batch_size_per_gpu": "auto", "wall_clock_breakdown": false } |
| GPU Utilization (%)      | 前两个实验的核心在计算时都被充分利用起来，后两个实验有空闲                    | ![](https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/a35e4b300d6847bb9ac3cf79c8d0551e~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAg5oiR5piv546L5aSn5L2g5piv6LCB:q75.awebp?policy=eyJ2bSI6MywidWlkIjoiNTM2MjE3NDA1ODk1MTQ5In0%3D&rk3s=e9ecf3d6&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1741158689&x-orig-sign=%2BqeZ5BDHjKB0pGXd%2BoJ77ODfbI4%3D)                                                                                                                                                                                                                                                                                                                                      | ![](https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/d6843800ebdf414484d6f9d8e0b1d2fa~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAg5oiR5piv546L5aSn5L2g5piv6LCB:q75.awebp?policy=eyJ2bSI6MywidWlkIjoiNTM2MjE3NDA1ODk1MTQ5In0%3D&rk3s=e9ecf3d6&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1741158689&x-orig-sign=NMhyQnBTX8S00ca5HBq4EUOlj7A%3D)                                                                                                                                                                                                                                                                                                                                        | ![](https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/f9d410a89a4a4490a80e07b04c02c915~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAg5oiR5piv546L5aSn5L2g5piv6LCB:q75.awebp?policy=eyJ2bSI6MywidWlkIjoiNTM2MjE3NDA1ODk1MTQ5In0%3D&rk3s=e9ecf3d6&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1741158689&x-orig-sign=zgY5iDzWleLyzL12w6CPJK0eEJg%3D)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        | ![](https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/583b3e76997543619200427c7f8b9373~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAg5oiR5piv546L5aSn5L2g5piv6LCB:q75.awebp?policy=eyJ2bSI6MywidWlkIjoiNTM2MjE3NDA1ODk1MTQ5In0%3D&rk3s=e9ecf3d6&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1741158689&x-orig-sign=BoqycUy0TtXPolEpQV730J%2FMoPs%3D)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| GPU Memory Allocated (%) | stage3 消耗显存最少，也就是节省显存最多                          | ![](https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/68986a39cbe141338ddf66585a3f8994~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAg5oiR5piv546L5aSn5L2g5piv6LCB:q75.awebp?policy=eyJ2bSI6MywidWlkIjoiNTM2MjE3NDA1ODk1MTQ5In0%3D&rk3s=e9ecf3d6&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1741158689&x-orig-sign=vVLdP%2FKLVuuT3XjgtS5yLiqrh%2F4%3D)                                                                                                                                                                                                                                                                                                                                      | ![](https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/12b7f1b755494571bb5842127492ef57~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAg5oiR5piv546L5aSn5L2g5piv6LCB:q75.awebp?policy=eyJ2bSI6MywidWlkIjoiNTM2MjE3NDA1ODk1MTQ5In0%3D&rk3s=e9ecf3d6&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1741158690&x-orig-sign=eIroBTo4iM4Y1BOkL%2BfBOvgQv8E%3D)                                                                                                                                                                                                                                                                                                                                        | ![](https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/b1384ebfb77d4a80a9cbfd56d02dd2a0~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAg5oiR5piv546L5aSn5L2g5piv6LCB:q75.awebp?policy=eyJ2bSI6MywidWlkIjoiNTM2MjE3NDA1ODk1MTQ5In0%3D&rk3s=e9ecf3d6&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1741158689&x-orig-sign=EfHjkOIcAiXlyPBEfB0qPFg7zwY%3D)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        | ![](https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/ab9dea6fd3c5416e99387e2a1242eced~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAg5oiR5piv546L5aSn5L2g5piv6LCB:q75.awebp?policy=eyJ2bSI6MywidWlkIjoiNTM2MjE3NDA1ODk1MTQ5In0%3D&rk3s=e9ecf3d6&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1741158689&x-orig-sign=Mlon%2BawJfY1jVX4vYwujquPWLM8%3D)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| CPU Utilization (%)      | CPU 利用率逐渐上升，因为后面两个实验需要数据在 CPU 和 GPU 之间频繁传输       | ![](https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/e18b7eb9ee7a43d1af9e2c50ea3a7cd7~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAg5oiR5piv546L5aSn5L2g5piv6LCB:q75.awebp?policy=eyJ2bSI6MywidWlkIjoiNTM2MjE3NDA1ODk1MTQ5In0%3D&rk3s=e9ecf3d6&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1741158689&x-orig-sign=Un0Fo0gbK3W9%2B0wM1IaiZfYevdU%3D)CPU Utilization (%) 7.6                                                                                                                                                                                                                                                                                                               | ![](https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/755665724cad48f3a20484c4718cf9b9~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAg5oiR5piv546L5aSn5L2g5piv6LCB:q75.awebp?policy=eyJ2bSI6MywidWlkIjoiNTM2MjE3NDA1ODk1MTQ5In0%3D&rk3s=e9ecf3d6&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1741158689&x-orig-sign=1dWnU8Ml%2BkvOTbcvte9RC6vf8eM%3D)CPU Utilization (%) 8.5                                                                                                                                                                                                                                                                                                                 | ![](https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/3493bdbbef094533895179d3551aa6a5~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAg5oiR5piv546L5aSn5L2g5piv6LCB:q75.awebp?policy=eyJ2bSI6MywidWlkIjoiNTM2MjE3NDA1ODk1MTQ5In0%3D&rk3s=e9ecf3d6&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1741158689&x-orig-sign=fOos1hZC1WSGLwUv2XHJ5KGEZGw%3D)CPU Utilization (%) 9                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   | ![](https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/60c51cdeecad414b8f4d121fc5a90a8d~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAg5oiR5piv546L5aSn5L2g5piv6LCB:q75.awebp?policy=eyJ2bSI6MywidWlkIjoiNTM2MjE3NDA1ODk1MTQ5In0%3D&rk3s=e9ecf3d6&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1741158689&x-orig-sign=7sBDMSvVvQe7NM70lGQ%2FpvnJvEQ%3D)CPU Utilization (%) 11.5                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| 内存消耗                     | 前两个大部分时间的内存消耗差不多，后两个内存消耗大幅上升，因为参数和优化器被卸载到 CPU 内存 | ![](https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/ebf50221d33d4d168c4271d204643667~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAg5oiR5piv546L5aSn5L2g5piv6LCB:q75.awebp?policy=eyJ2bSI6MywidWlkIjoiNTM2MjE3NDA1ODk1MTQ5In0%3D&rk3s=e9ecf3d6&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1741158689&x-orig-sign=MbMDQfhbjH60%2BWw2VzOEsvwAlVU%3D)Process Memory In Use (MB) 2430.9609375![](https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/99d9c7b22c29428e94797664b5ac1794~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAg5oiR5piv546L5aSn5L2g5piv6LCB:q75.awebp?policy=eyJ2bSI6MywidWlkIjoiNTM2MjE3NDA1ODk1MTQ5In0%3D&rk3s=e9ecf3d6&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1741158689&x-orig-sign=q3tIEWnO%2Bv0s%2B8GasMaCuOne6WA%3D)Process Memory In Use (MB) 28622.5234375 | ![](https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/40d2bc4c1d2540b9bef4a0f40950f624~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAg5oiR5piv546L5aSn5L2g5piv6LCB:q75.awebp?policy=eyJ2bSI6MywidWlkIjoiNTM2MjE3NDA1ODk1MTQ5In0%3D&rk3s=e9ecf3d6&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1741158689&x-orig-sign=rpPo6%2FAyvy%2FXwnihm9YsoGw056U%3D)Process Memory In Use (MB) 2421.48828125![](https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/385915a29ff647d0b23fe375540ca6d5~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAg5oiR5piv546L5aSn5L2g5piv6LCB:q75.awebp?policy=eyJ2bSI6MywidWlkIjoiNTM2MjE3NDA1ODk1MTQ5In0%3D&rk3s=e9ecf3d6&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1741158689&x-orig-sign=pdJKCziPTIz0FG%2BMc7TTZtw7sgE%3D)Process Memory In Use (MB) 27809.80078125 | ![](https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/8839383e78eb4da8a6c79235dd16285c~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAg5oiR5piv546L5aSn5L2g5piv6LCB:q75.awebp?policy=eyJ2bSI6MywidWlkIjoiNTM2MjE3NDA1ODk1MTQ5In0%3D&rk3s=e9ecf3d6&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1741158689&x-orig-sign=tn%2FcXzT5UYB%2B7FbprnIJ1fElCAM%3D)Process Memory In Use (non-swap) (MB) 2679.4375![](https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/49b95d1f321f4f4595fadfe55c56fc19~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAg5oiR5piv546L5aSn5L2g5piv6LCB:q75.awebp?policy=eyJ2bSI6MywidWlkIjoiNTM2MjE3NDA1ODk1MTQ5In0%3D&rk3s=e9ecf3d6&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1741158689&x-orig-sign=SrdP2aBwI7y03CKhm5MF4iKwaGs%3D)Process Memory In Use (MB) 25693.34765625                                                                                                                                                                                                                                                                          | ![](https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/8f6ec9025ce24f3982cd9babbffbf994~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAg5oiR5piv546L5aSn5L2g5piv6LCB:q75.awebp?policy=eyJ2bSI6MywidWlkIjoiNTM2MjE3NDA1ODk1MTQ5In0%3D&rk3s=e9ecf3d6&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1741158689&x-orig-sign=9uxeo6AKxjMkKl5fYssIB21MMK0%3D)Process Memory In Use (MB) 27213.0546875![](https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/18971d6a4b8c4c79bc2f0a7d9694de0f~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAg5oiR5piv546L5aSn5L2g5piv6LCB:q75.awebp?policy=eyJ2bSI6MywidWlkIjoiNTM2MjE3NDA1ODk1MTQ5In0%3D&rk3s=e9ecf3d6&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1741158689&x-orig-sign=AMgGUQFJIYyqt3b6M6aeMy7vrEw%3D)Process Memory In Use (MB) 49028.27734375