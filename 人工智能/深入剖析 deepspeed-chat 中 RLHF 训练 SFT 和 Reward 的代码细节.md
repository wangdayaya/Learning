# 前文

我对 RLHF 的具体细节一直模棱两可，所以干脆找了微软的项目 [DeepSpeed-Chat](https://github.com/microsoft/DeepSpeedExamples/tree/master/applications/DeepSpeed-Chat) 中的具体实现的代码拿来研究细节，`我参考了几位大佬的文章，尽量用自己最简洁的语言介绍清楚，我这里只抓核心代码，无关紧要的代码都被我删掉了，不影响整体逻辑，详细的解释都在代码前后注释了`。

# 准备

## 基本知识

本文只讲 PPO 具体代码细节，我这里不会详细解释大语言模型基础和强化学习的入门概念，所以需要您自行学习准备，如：

*   Transformer
*   LLM
*   RLHF
*   动作价值函数
*   状态价值函数
*   最优动作价值函数
*   最优状态价值函数
*   折扣回报
*   奖励
*   动作
*   状态
*   策略
*   优势函数
*   PPO 算法
*   ...

## 模型

    git clone https://huggingface.co/facebook/opt-350m
    git clone https://huggingface.co/facebook/opt-1.3b

下载好之后，将各自目录下的 `config.json`中的`_name_or_path`改为本地模型路径。第一个模型当作后续的 actor ，第二个模型当作 reward 。

## 数据

    git clone https://huggingface.co/datasets/Dahoas/rm-static
    git clone https://huggingface.co/datasets/Dahoas/full-hh-rlhf
    git clone https://huggingface.co/datasets/Dahoas/synthetic-instruct-gptj-pairwise
    git clone https://huggingface.co/datasets/yitingxie/rlhf-reward-datasets

将这些数据的提示词、response、chosen 、rejected 都整理成以下的格式，也就是使用“Human:”进行查询，使用“Assistant:”进行回答的统一数据形式，其中的 response 和 chosen 是一样的。方便后面使用。官方的[数据处理](https://github.com/microsoft/DeepSpeedExamples/blob/master/applications/DeepSpeed-Chat/dschat/utils/data/raw_datasets.py)代码已经帮我们实现好了，我们只需要把里面的各个数据在本地的位置修改一下即可。以下面的例子的形式为准。

| prompt (string)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               | response (string)                                                       | chosen (string)                                                         | rejected (string)                                                                               |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------- | ----------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------- |
| " Human: I am trying to write a fairy tale. What is the most popular plot? Assistant: The most popular plot might be “a princess goes to a faraway land, falls in love, and has a magic genie that grants her wishes”. We can find plenty of examples of this if we search for fairy tales using the search engine Google. Human: You can't look anything up on google. Assistant: OK, so let’s think about some other popular fairy tales. Human: Do you like the plot of Hansel and Gretel? ... Assistant:" | " This sounds like a really interesting modern retelling of the story!" | " This sounds like a really interesting modern retelling of the story!" | " And the prince and the princess both decide that they are more powerful together than apart?" |

# 训练模式和推理模式

`LLM` 的训练模式使用 `teacher force` 的方式，将整句话输入到模型中，并通过 `mask` 机制在保证不泄漏未来的单词情况下预测下一个单词。

`LLM` 的推理模式是真正的自回归，预测出下一个单词之后，当作下一步输入再预测下下一个单词。

![](https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/d104d12edbf24bed825f161773d99c3c~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAg5oiR5piv546L5aSn5L2g5piv6LCB:q75.awebp?policy=eyJ2bSI6MywidWlkIjoiNTM2MjE3NDA1ODk1MTQ5In0%3D&rk3s=f64ab15b&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1733205836&x-orig-sign=TOhIKQvmAn%2FHv8mP%2BAtj1asLJQM%3D)

# Step 1：SFT

## 制作训练数据

1.  选取一条样本数据的 prompt 和 chosen ，直接将他们进行拼接，最后加上一个终止符 token （是 </s> ），形成一条完整的训练数据文本 chosen\_sentence 。
2.  使用 tokenizer 对 chosen\_sentence 进行分词，同时按照 max\_length 进行截断或者 padding 填充（这里的 padding 对应的 token 是也是 </s> ，不过不影响结果），得到分词张量 chosen\_token 。这里使用的 tokenzier 是从本地模型 opt-1.3b 处加载的。

<!---->

    chosen_sentence = raw_dataset.get_prompt_and_chosen(tmp_data)

    if chosen_sentence is not None:

    chosen_sentence += end_of_conversation_token

    chosen_token = tokenizer(chosen_sentence, max_length=max_seq_len, padding="max_length", truncation=True, return_tensors="pt")

3.  最终，一条样本制作之后返回的结果是 chosen\_sentence 分词结果中的 input\_ids 、 attention\_mask 、lables ，这里有个比较有意思的点，input\_ids 和 labels 的有效部分是一样的，也就是用相同的输入通过模型得到相同的标签输出，之前 mask 为 0 的部分都设置为 -100 。

<!---->

    return {

    "input_ids": self.chosen_dataset[idx]["input_ids"],

    "attention_mask": self.chosen_dataset[idx]["attention_mask"],

    "labels": torch.where(self.chosen_dataset[idx]["attention_mask"].bool(), self.chosen_dataset[idx]["input_ids"], -100)

    }

## 训练模型

使用下载好的 opt-1.3b 模型进行常规的语言模型训练即可。可以参考[全量参数和LoRA部分参数微调qwen模型](https://hl8xut0wpb.feishu.cn/docx/HH3VdWLFwopQfLxRTCLcHlJtnlb#share-Gq9KdBb8moynLCxfzBvcoQzNnog) ，只不过训练数据的输入和标签的形式不同，但原理是一样的。只不过我们这里需要对问题和答案进行综合考虑来优化模型，所以会将问题和答案都作为标签返回，如果只需要返回答案，只需要按照[全量参数和LoRA部分参数微调qwen模型](https://hl8xut0wpb.feishu.cn/docx/HH3VdWLFwopQfLxRTCLcHlJtnlb#share-Gq9KdBb8moynLCxfzBvcoQzNnog) 里面的训练数据进行组织即可。

# Step 2：奖励模型训练

## 架构图

![](https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/c25ec1f31e4b4047a82080664eb96020~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAg5oiR5piv546L5aSn5L2g5piv6LCB:q75.awebp?policy=eyJ2bSI6MywidWlkIjoiNTM2MjE3NDA1ODk1MTQ5In0%3D&rk3s=f64ab15b&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1733205836&x-orig-sign=pXMKING5LTuSroBJRWyCUatCz2c%3D)

我们可以回看源代码中的[奖励模型](https://github.com/microsoft/DeepSpeedExamples/blob/master/applications/DeepSpeed-Chat/dschat/utils/model/reward_model.py)的具体实现。需要注意的是这份奖励模型代码后面在 `PPO 算法优化阶段`被使用到，创建出两个一样的副本模型 `reward_model 和 critic_model` ，只是它们的所输出的值有所不同。

## 制作训练数据

1.  选取一条样本数据的 prompt 和 chosen ，直接将他们进行拼接，最后加上一个终止符 token 是 </s> ，形成一条完整的训练数据文本 chosen\_sentence 。同样的再用其中的 prompt 和 rejected 进行拼接，最后最后加上一个终止符 token 是 </s> ，形成一条完整的训练数据文本 reject\_sentence 。
2.  使用 tokenizer 分别对对 chosen\_sentence 和 reject\_sentence 进行分词，同时按照 max\_length 进行截断或者 padding 填充（这里的 padding 对应的 token 是也是 </s> ，不过不影响结果），得到分词张量 chosen\_token 和 reject\_token 。这里使用 的 tokenizer 是本地下载好的 opt-350m 处加载来的。

```


chosen_sentence = raw_dataset.get_prompt_and_chosen( tmp_data)

reject_sentence = raw_dataset.get_prompt_and_rejected(tmp_data)

if chosen_sentence is not None and reject_sentence is not None:

chosen_sentence += end_of_conversation_token # the accept response

reject_sentence += end_of_conversation_token

chosen_token = tokenizer(chosen_sentence, max_length=max_seq_len, padding="max_length", truncation=True, return_tensors="pt")

reject_token = tokenizer(reject_sentence, max_length=max_seq_len, padding="max_length", truncation=True, return_tensors="pt")
```

3.  最终，一条样本制作之后返回的结果是 chosen\_sentence 和 reject\_sentence 分词结果中的的 input\_ids 、 attention\_mask。

<!---->

    return self.chosen_dataset[idx]["input_ids"], self.chosen_dataset[idx]["attention_mask"], self.reject_dataset[idx]["input_ids"], self.reject_dataset[idx]["attention_mask"]

4.  在最后使用的时候，要进行整理，将所有的正面回答样本和负面回答样本按照 batch 维度都汇聚成完整的一个 batch 的 input\_ids 和 attention\_mask ，在训练使用的时候还分再分开处理的，可以继续看下一节的训练介绍。

<!---->

    class DataCollatorReward:

    def __call__(self, data):

    batch = {}

    batch["input_ids"] = torch.cat([f[0] for f in data] + [f[2] for f in data], dim=0)

    batch["attention_mask"] = torch.cat([f[1] for f in data] + [f[3] for f in data], dim=0)

    return batch

## 训练

模型训练的代码可以查看[奖励模型源代码](https://github.com/microsoft/DeepSpeedExamples/blob/master/applications/DeepSpeed-Chat/dschat/utils/model/reward_model.py)中的 forward 函数，输入的 input\_ids 和 attention\_mask 中前一半是正面问答，后一半是负面问答，所以先要对半处理一下。

    def forward(self, input_ids=None, past_key_values=None, attention_mask=None,  position_ids=None, head_mask=None, inputs_embeds=None, use_cache=False):
        ...
        # 奖励模型其实就是一个 opt-350m + 一个额外定义的线性层可以将模型的输出转为一个 scalar 
        transformer_outputs = self.rwtransformer( input_ids, past_key_values=past_key_values,  attention_mask=attention_mask, inputs_embeds=inputs_embeds,  use_cache=use_cache, **kwargs)
        hidden_states = transformer_outputs[0] # (batch_size, seq_len, hidden_size)
        #  v_head 就是一个线性层 nn.Linear(self.config.n_embd, 1, bias=False)
        rewards = self.v_head(hidden_states).squeeze(-1)  # (batch_size, seq_len, 1)  ->  (batch_size, seq_len) 
        
        chosen_mean_scores = []
        rejected_mean_scores = []

        # 将 inputs 和 rewards 切分成  chosen 和 rejected 两部分，原因在上面已经解释了
        assert len(input_ids.shape) == 2
        bs = input_ids.shape[0] // 2
        seq_len = input_ids.shape[1]
        chosen_ids = input_ids[:bs]   # (bs, seq_len)
        rejected_ids = input_ids[bs:]  # (bs, seq_len)
        chosen_rewards = rewards[:bs]  # (bs, seq_len)
        rejected_rewards = rewards[bs:]  # (bs, seq_len)

        # Compute pairwise loss. Only backprop on the different tokens before padding
        loss = 0.
        for i in range(bs):
            chosen_id = chosen_ids[i]
            rejected_id = rejected_ids[i]
            chosen_reward = chosen_rewards[i]
            rejected_reward = rejected_rewards[i]
            c_inds = (chosen_id == self.PAD_ID).nonzero()
            # 确定 chosen_id 中有效 token 的末尾位置，适配不同模型的填充策略，如 OPT 模型可能在序列开头添加填充值。
            c_ind = c_inds[self.num_padding_at_beginning].item() if len(  c_inds ) > self.num_padding_at_beginning else seq_len  # OPT model pads the first token, so we need to use the second padding token as the end of the sequence
            # 找出 chosen_id 和 rejected_id 中第一个不同的 token 索引。它们的问题都一样，不一样的地方肯定是答案开始的位置
            check_divergence = (chosen_id != rejected_id).nonzero()
            # 如果两个序列完全相同
            if len(check_divergence) == 0:
                end_ind = rejected_reward.size(-1)
                divergence_ind = end_ind - 1
                r_ind = c_ind
            else:
                r_inds = (rejected_id == self.PAD_ID).nonzero()
                # 找出 rejected 答案中的最后一个有效 token 位置
                r_ind = r_inds[self.num_padding_at_beginning].item() if len(r_inds) > self.num_padding_at_beginning else seq_len
                end_ind = max(c_ind, r_ind)
                divergence_ind = check_divergence[0]
            assert divergence_ind > 0
            # 从 divergence_ind 到 end_ind 截取 chosen_reward 和 rejected_reward 的有效部分。
            # divergence_ind 是两个序列不同的第一个位置索引，end_ind 是两个序列中有效 token 位置索引的较大值 
            c_truncated_reward = chosen_reward[divergence_ind:end_ind]  # (有效 token 数,)
            r_truncated_reward = rejected_reward[divergence_ind:end_ind]  # (有效 token 数,)
            # 将 chosen_reward 和 rejected_reward 的末尾有效 token 的奖励值添加到对应的列表中，用于计算最终的平均分数。
            chosen_mean_scores.append(chosen_reward[c_ind - 1])  
            rejected_mean_scores.append(rejected_reward[r_ind - 1])
            ...
            # 截取后的 c_truncated_reward 和 r_truncated_reward 相减，序列中每个 token 是奖励值。取 mean 是为了让损失与序列长度无关，保证梯度对不同样本的贡献一致性
            loss += -torch.nn.functional.logsigmoid(c_truncated_reward - r_truncated_reward).mean()

        loss = loss / bs
        chosen_mean_scores = torch.stack(chosen_mean_scores)
        rejected_mean_scores = torch.stack(rejected_mean_scores)
        return {
            "loss": loss,  # 损失值
            "chosen_mean_scores": chosen_mean_scores,   # （B，）  里面是正面答案的分值
            "rejected_mean_scores": rejected_mean_scores,   #（B，）  里面是负面答案的分值
        }

优化采用 `pair wiss loss`，即输入模型关于同一个问题的两个回答，让模型学会这两个句子哪个分高哪个分低。`pair wise loss` 代码如下，目标就是尽量给 `pair` 里边好的答案打分高（c\_truncated\_reward），给差的答案（r\_truncated\_reward）打分低，两个答案的 token 序列进行相减，比较每个对应位置的得分差距，分值差距越大， loss 才能变小，关键代码如下：

    c_truncated_reward = chosen_reward[divergence_ind:end_ind]
    r_truncated_reward = rejected_reward[divergence_ind:end_ind]
    ...
    loss += -torch.nn.functional.logsigmoid(c_truncated_reward - r_truncated_reward).mean()

## 计算奖励值

奖励模型的输出可以查看[奖励模型源代码](https://github.com/microsoft/DeepSpeedExamples/blob/master/applications/DeepSpeed-Chat/dschat/utils/model/reward_model.py)中的 `forward_value`函数，经过训练，奖励模型的输入是 `prompt+answer+pad` 的形式，让模型学会对 `prompt+answer` 进行打分即可，最后返回的是一个 `scalar` 。

    def forward_value(self,  input_ids=None,  attention_mask=None, past_key_values=None,  position_ids=None,  head_mask=None, inputs_embeds=None, return_value_only=False, prompt_length=0,  use_cache=False):
        ...
        transformer_outputs = self.rwtransformer(input_ids, past_key_values=past_key_values, attention_mask=attention_mask,  inputs_embeds=inputs_embeds, use_cache=use_cache, **kwargs)
        hidden_states = transformer_outputs[0]
        #  v_head 就是一个线性层 nn.Linear(self.config.n_embd, 1, bias=False)
        # 表示每个 token 在序列中的 reward 值。
        values = self.v_head(hidden_states).squeeze(-1)  # (batch_size, seq_len, 1)  ->  (batch_size, seq_len) 
        if return_value_only:
            return values  # (batch_size, seq_len) 
        else:
            assert prompt_length > 1, "prompt_length must be greater than 1 to help select the end score"
            bs = values.size(0)
            seq_len = input_ids.shape[1]
            chosen_end_scores = [ ]  
            for i in range(bs):
                input_id = input_ids[i]
                value = values[i]
                # 从第 prompt_length 个 token 开始，查找 padding tokens (PAD_ID) 的位置
                # 表示 padding token 的索引
                c_inds = (input_id[prompt_length:] == self.PAD_ID).nonzero()
                # 如果存在 padding token，c_ind 为第一个 padding token 的索引（加上 prompt_length 偏移量后）。
                # c_ind 是一个整数，表示序列有效结束的位置。
                c_ind = c_inds[0].item() + prompt_length if len( c_inds) > 0 else seq_len
                # 将序列有效结束位置（c_ind - 1）的 reward 值添加到 chosen_end_scores 中
                chosen_end_scores.append(value[c_ind - 1])
            return {
                "values": values,  # (batch_size, seq_len)
                "chosen_end_scores": torch.stack(chosen_end_scores),  # (batch_size,)  每个序列中有效的答案序列的最后一个位置的值作为分数
            }

1.  **结果输出的字典中两部分内容详细解释**

*   `values` 维度为 `(B,L)` ，序列中个每个值都对应了每个 token 的奖励值。但是每一个有效 token 的奖励值，尤其是**最后一个有效 token 的奖励值，** 因为间接隐含累积了前面整个序列的质量好坏信息，所以还能被代表其前面序列的整体奖励值（和 `DQN` 里边的 `Q` 值一个意义， Q(s, a) 表示在给定状态 ( s ) 下，采取动作 ( a ) 并遵循某策略后，智能体从当前状态到未来状态所能获得的预期总回报），这个输出满足了后续 `PPO` 算法优化阶段 `critic_model` 的输出要求。

*   `chosen_end_scores` 表示的是奖励模型对于 `prompt` 做出的`answer` 的评分，这个输出满足了后续 `PPO` 算法优化阶段 `reward_model` 的输出要求。其实 `chosen_end_scores` 也是通过将 `values` 经过处理拿到的。`chosen_end_scores` 列表包含了每个模型输出答案的评分，这些分数是基于序列的最后一个有效标记（不是填充标记）的值，这个值可以用于评估模型生成的答案质量，维度是 `(B,)` 。

2.  **为什么不用所有 token 的奖励值计算整体评分？**

虽然可以使用所有 token 的奖励值（如求和、平均等）作为整体评分，但这并不常见，原因是：

*   **生成任务的评价逻辑：** 任务通常关心的是整个序列的生成质量，而最后一个有效 token 的奖励值最能代表模型对整个序列的最终评价。

*   **避免稀释信息：** 如果用所有 token 的奖励值计算整体评分，可能会稀释高质量 token 的信息。特别是对较长的序列，填充 token 或中间 token 的低质量评分会影响整体评价。

3.  **最后一个有效 token 的奖励值表示整体评分的原因**

在序列生成任务中，**序列中的奖励值**通常是用最后一个有效 token 的奖励值来表示，原因如下：

*   **自然语言生成的因果依赖性：** 生成模型（如 GPT 或 LLaMA）是自回归模型，生成序列时，前面的 token 会影响后面的 token，但后面的 token 不会影响前面的 token。因此，最后一个有效 token 已经累积了整个序列的上下文信息，能代表生成序列的整体质量。

*   **奖励模型的作用:** 奖励模型的目标是给生成的序列打分，而不是评估每个 token 的奖励值。逐位置的奖励值只是一个中间产物。序列的最终质量可以通过最后一个有效 token 的奖励值来衡量。

4.  **序列奖励值示例**

假设模型的输入序列如下，序列中有问题、答案、pad 的 token ：

    input_ids = [[101, 2009, 2003, 103, 0, 0],  # 第一行有填充
                 [101, 1996, 3075, 102, 0, 0]]  # 第二行也有填充

*   `0` 是 `<PAD>`，`103`、102 是第一个序列的最后一个有效 token。
*   模型输出的 `values` 是：

<!---->

    values = [[0.1, 0.3, 0.5, 0.7, -0.2, -0.2],
              [0.2, 0.4, 0.6, 0.8, -0.1, -0.1]]

*   对于第一行，`values[0][3] = 0.7` 是最后一个有效 token 的奖励值，用来表示整个序列的评分。

*   对于第二行，`values[1][3] = 0.8` 是最后一个有效 token 的奖励值，用来表示整个序列的评分。
# Step 3： PPO 

请移步阅读下一篇文章：https://juejin.cn/post/7441238206740103178

# 参考

*   <https://zhuanlan.zhihu.com/p/624589622>
*   <https://huggingface.co/blog/rlhf>
*   <https://mathmach.com/be7f3b4f/>
*   <https://github.com/microsoft/DeepSpeedExamples/blob/master/applications/DeepSpeed-Chat/dschat/utils/model/reward_model.py>
