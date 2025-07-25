# 数据

自己制作的数据，训练数据 2700 条，测试数据 290 条，格式如下：

```
{"instruction": "2631", "input": "", "output": "王大丫丫的回答是--->2164 + 467"}
{"instruction": "2632", "input": "", "output": "王大丫丫的回答是--->2107 + 525"}
{"instruction": "2633", "input": "", "output": "王大丫丫的回答是--->1586 + 1047"}
{"instruction": "2634", "input": "", "output": "王大丫丫的回答是--->1038 + 1596"}
{"instruction": "2635", "input": "", "output": "王大丫丫的回答是--->1456 + 1179"}
{"instruction": "2636", "input": "", "output": "王大丫丫的回答是--->1434 + 1202"}
```

# 硬件

1 张 4090

# 预先查看 QWEN2.5-3B 在本测试数据的效果

日志如下，准确率为 0 ：

```
输入：2710 输出：您提供的数字2710似乎是一串数字，但没有明确的数学问题或运算符。如果您需要对这个数字进行某种计算
输入：2711 输出：您提供的数字2711似乎是一个数值，但并没有给出具体的问题。请问您是想要对这个数字进行某种运算，还是说
输入：2712 输出：您提供的数字2712是一个具体的数值，但并没有给出任何问题或需要进行的操作。如果您是想询问有关这个数字的问题，
输入：2713 输出：您提供的数字2713是一个具体的数值，它本身并不构成一个明确的数学问题。如果您需要对这个数字进行某种操作或
输入：2714 输出：您提供的数字2714似乎是一串数字，而不是一个完整的数学问题。如果您需要对这个数字进行某种操作或解决某个与
输入：2715 输出：您提供的数字2715是一个具体的数值，它本身并没有一个明确的问题需要解决。不过，如果您有特定的问题或求解的目标
输入：2716 输出：您似乎提供了一个数值，但是没有给出具体的问题。2716是一个数字，它可以是任何数学问题的答案，比如一个乘法
输入：2717 输出：您提供的数字2717似乎是一个数值，但没有给出具体的问题情境。因此，我无法确定您需要什么样的解答。如果您能
...
0.0
```

可以看出主要是因为模型对于我的测试集输入不明白要干嘛。

# 第一步 LORA SFT QWEN2.5-3B

## swanlab

https://swanlab.cn/@wangdayaya/Distillation/runs/h331r8b44pu5lf5fui827/chart

耗时1分钟20秒

## 效果

日志打印如下：

```
输入：710  输出：王大丫丫的回答是--->694 + 2016
输入：2711 输出：王大丫丫的回答是--->2039 + 672
输入：2712 输出：王大丫丫的回答是--->586 + 2126
输入：2713 输出：王大丫丫的回答是--->2689 + 24
输入：2714 输出：王大丫丫的回答是--->205 + 2509
输入：2715 输出：王大丫丫的回答是--->490 + 2225
输入：2716 输出：王大丫丫的回答是--->380 + 2336
输入：2717 输出：王大丫丫的回答是--->183 + 2534
输入：2718 输出：王大丫丫的回答是--->439 + 2279
输入：2719 输出：王大丫丫的回答是--->206 + 2513
...
1.0
```

可以看出模型已经学会了如何去处理我的输入，并且给出合理的答案，最终的正确率可以达到 100% 。

  


# 第二步 QWEN2.5-3B 合并 LORA

最终生成了 3B_lora_merge 模型。

```
from transformers import AutoModelForCausalLM
from peft import PeftModel
from transformers import AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(r"D:\Qwen2.5-3B-Instruct")
model = PeftModel.from_pretrained(model, r"D:\PycharmProjects\Distillation\lora_3B\checkpoint-338")
merged_model = model.merge_and_unload()

# 把合并后的模型保存到指定的目录
merged_model.save_pretrained("3B_lora_merge", max_shard_size="2048MB", safe_serialization=True)
tokenizer = AutoTokenizer.from_pretrained(r"D:\Qwen2.5-3B-Instruct")
tokenizer.save_pretrained("3B_lora_merge")
```

  


# 第三步直接使用 kl 散度进行 LORA 蒸馏 QWEN2.5-0.5B

## 核心代码

主要是改一下 Trainer 的计算 loss 的实现，改为 kl 散度计算损失。

```

def compute_fkl(logits, teacher_logits, target, padding_id, reduction="sum", temp=1.0):
    logits = logits / temp
    teacher_logits = teacher_logits / temp
    log_probs = torch.log_softmax(logits, -1, dtype=torch.float32)
    teacher_probs = torch.softmax(teacher_logits, -1, dtype=torch.float32)
    teacher_log_probs = torch.log_softmax(teacher_logits, -1, dtype=torch.float32)
    kl = (teacher_probs * (teacher_log_probs - log_probs))
    kl = kl.sum(-1)
    if reduction == "sum":
        pad_mask = target.eq(padding_id)
        kl = kl.masked_fill_(pad_mask, 0.0)
        kl = kl.sum()
    return kl


class KGTrainer(Trainer):
    def __init__(
            self,
            model=None,
            teacher_model=None,
            if_use_entropy=False,
            args=None,
            data_collator=None,
            train_dataset=None,
            eval_dataset=None,
            tokenizer=None,
            model_init=None,
            compute_metrics=None,
            callbacks=None,
            preprocess_logits_for_metrics=None,
    ):
        super().__init__(
            model,
            args,
            data_collator,
            train_dataset,
            eval_dataset,
            tokenizer,
            model_init,
            compute_metrics,
            callbacks,
            preprocess_logits_for_metrics,
        )
        self.teacher_model = teacher_model
        self.if_use_entropy = if_use_entropy

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        outputs = model(**inputs)
        with torch.no_grad():
            teacher_outputs = self.teacher_model(**inputs)
        loss = outputs.loss
        logits = outputs.logits
        teacher_logits = teacher_outputs.logits
        if logits.shape[-1] != teacher_logits.shape[-1]:
            teacher_logits = teacher_logits[:, :, :logits.shape[-1]]
        labels = inputs['labels']
        kl = compute_fkl(logits, teacher_logits, labels, padding_id=-100, temp=2.0)
        if self.if_use_entropy:  # 不微调学生模型的时候，kl 散度损失和交叉熵损失加权
            loss_total = 0.5 * kl + 0.5 * loss
        else:
            loss_total = kl
        return (loss_total, outputs) if return_outputs else loss_total
```

## swanlab

https://swanlab.cn/@wangdayaya/Distillation/runs/bj4yjem2w5fpap6xh45xh/chart

耗时 8分钟57秒

## 效果

```
输入：2710 输出：王大丫丫的回答是--->120 + 2690
输入：2711 输出：王大丫丫的回答是--->1044 + 1667
输入：2712 输出：王大丫丫的回答是--->1208 + 1504
输入：2713 输出：王大丫丫的回答是--->1211 + 1502
输入：2714 输出：王大丫丫的回答是--->104 + 2600
输入：2715 输出：王大丫丫的回答是--->108 + 2607
输入：2716 输出：王大丫丫的回答是--->120 + 2696
输入：2717 输出：王大丫丫的回答是--->111 + 2606
输入：2718 输出：王大丫丫的回答是--->120 + 2698
输入：2719 输出：王大丫丫的回答是--->1214 + 1505
...
生成错误 输入：2714 输出：王大丫丫的回答是--->104 + 2600
生成错误 输入：2710 输出：王大丫丫的回答是--->120 + 2690
生成错误 输入：2716 输出：王大丫丫的回答是--->120 + 2696
生成错误 输入：2718 输出：王大丫丫的回答是--->120 + 2698
生成错误 输入：2724 输出：王大丫丫的回答是--->104 + 2610
生成错误 输入：2823 输出：王大丫丫的回答是--->1227 + 1606
生成错误 输入：2881 输出：王大丫丫的回答是--->121 + 2750
生成错误 输入：2905 输出：王大丫丫的回答是--->108 + 2807

0.9724137931034482
```

准确率达到了 97.24% ，可以看出来蒸馏的效果还是不错的，学会了如何完成我的任务，尽管还有很小概率的生成错误。