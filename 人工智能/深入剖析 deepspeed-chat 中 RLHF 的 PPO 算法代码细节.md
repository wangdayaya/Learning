下面介绍的代码中会将不重要的代码删减，不影响算法逻辑，具体源代码实现可以回看 [ppo\_trainer.py](https://github.com/microsoft/DeepSpeedExamples/blob/master/applications/DeepSpeed-Chat/dschat/rlhf/ppo_trainer.py) 、[reward\_model.py](https://github.com/microsoft/DeepSpeedExamples/blob/master/applications/DeepSpeed-Chat/dschat/utils/model/reward_model.py)、[main.py](https://github.com/microsoft/DeepSpeedExamples/blob/master/applications/DeepSpeed-Chat/training/step3_rlhf_finetuning/main.py)。

# 准备
## RLHF 前两步训练

请移步复习上一篇文章：https://juejin.cn/post/7441215611589591077

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

# 架构图

*   整个 `PPO` 算法，涉及到`四个模型`。
*   `actor_model 和 ref_model` 一开始初始化都是第一阶段 `SFT` 训练好的模型的相同副本，只不过下面只更新 `actor_model`，但是强化学习过程很容易把模型训练“坏”，因此需要另外一个**参数不会更新或者某种策略慢更新**（如：直接 copy 或者 EMA 等策略）的 `ref_model` 来当作标的，别让 `actor mode` 跑偏太远，强化学习中防止自举现象。
*   `reward_model 和 critic_model` 一开始初始化都是第二阶段训练好的奖励模型的相同副本，只不过下面只更新 `critic_model` 。在这个图中没有画出来 `critic model` ，但是实际上在代码实现中是存在的，详见下面的初始化代码。
*   4 个模型共用的都是共用从 actor\_model\_name\_or\_path 加载来的 tokenizer 。

![](https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/5a20a42b72f54382841f7a75dd4288d8~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAg5oiR5piv546L5aSn5L2g5piv6LCB:q75.awebp?policy=eyJ2bSI6MywidWlkIjoiNTM2MjE3NDA1ODk1MTQ5In0%3D&rk3s=f64ab15b&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1733205911&x-orig-sign=11NYT%2BDdSUfjFnJtEjP%2B%2BQblwco%3D)

    class DeepSpeedRLHFEngine():
        def __init__(self, actor_model_name_or_path, critic_model_name_or_path,
                     tokenizer, args, num_total_iters):
            self.args = args
            self.num_total_iters = num_total_iters
            # 4 个模型共用的都是从 actor_model_name_or_path 加载来的 tokenizer
            self.tokenizer = tokenizer
            
            # actor 和 ref 都是从 actor_model_name_or_path 加载的
            self.actor = self._init_actor( actor_model_name_or_path=actor_model_name_or_path)
            self.ref = self._init_ref( actor_model_name_or_path=actor_model_name_or_path)
            self.actor_ema = None
            if self.args.enable_ema:
                self.actor_ema = self._init_ema( actor_model_name_or_path=actor_model_name_or_path)
            
            # critic 和 reward 都是从 critic_model_name_or_path 加载的，
            self.critic = self._init_critic( critic_model_name_or_path=critic_model_name_or_path)
            self.reward = self._init_reward( critic_model_name_or_path=critic_model_name_or_path)
            if self.args.critic_gradient_checkpointing:
                self.critic.gradient_checkpointing_enable()

# 训练数据制作

1.  只选取一条样本数据的 prompt 文本 。
2.  使用 tokenizer 分别对 prompt 进行分词，完全分词，不进行截断或者 padding 填充 ，得到分词张量 prompt\_token ，同时要将 prompt\_token\["input\_ids"] 和 prompt\_token\["attention\_mask"] 第 0 维去掉，并将序列值反转，后面会在第 4 步中再反转回来。这里需要注意已经特意按照 max\_seq\_len 进行了过滤，这里的 max\_seq\_len 的入参其实就是 max\_prompt\_seq\_len，默认为 256 。

<!---->

    prompt = raw_dataset.get_prompt(tmp_data)
    if prompt is not None:
        prompt_token = tokenizer(prompt, return_tensors="pt")
        if prompt_token["input_ids"].size()[-1] <= max_seq_len:
            # (1, seq_len) -> (seq_len) -> 反转，变成了 [1,2,3] - > [3,2,1]
            prompt_token["input_ids"] = prompt_token["input_ids"].squeeze(0).flip(0)
            # (1, seq_len) -> (seq_len) -> 反转，变成了 [1,1,1] - > [1,1,1]
            prompt_token["attention_mask"] = prompt_token["attention_mask"].squeeze(0).flip(0)

3.  最终，一条样本制作之后返回的结果是分词结果中的的 input\_ids 、 attention\_mask 、以及 tokenizer.pad\_token\_id 。

<!---->

    return self.prompt_dataset[idx]["input_ids"], self.prompt_dataset[idx]["attention_mask"], self.pad_token_id

4.  在最后使用的时候，要进行整理，将所有的训练数据都汇聚成完整的一个 batch 的。这里的 prompt 和 prompt\_att\_mask 最后要将所有的 pad 或者 0 都放到开头，这样模型生成内容可以紧接着 prompt token 继续推理。

<!---->

    class DataCollatorRLHF:
        def __init__(self, max_token_len, inference_tp_size):
            self.max_token_len = max_token_len  # max_token_len 用于指定最大 promt token 数量，默认为 256
            self.inference_tp_size = inference_tp_size  # 用于推理中的 tensor parallel 分块大小

        def __call__(self, data):
            batch = {}
            pad_token_id = data[-1][-1]  # 使用训练数据传过来的 tokenizer.pad_token_id
            # (batch_size, seq_len) ，使用 pad_token_id 末尾填充 input_ids 成相同长度的 batch 序列 ，会变成 [3,2,1,pad,pad,pad]
            prompt = pad_sequence([f[0] for f in data],  padding_value=pad_token_id, batch_first=True)
            # (batch_size, seq_len) ，使用 0 末尾填充 attention_mask 成相同长度的 batch 序列，会变成 [1,1,1,0,0,0]
            prompt_mask = pad_sequence([f[1] for f in data], padding_value=0,  batch_first=True)
            length = prompt.size()[-1]
            pad_length = self.max_token_len - length
            # 表示要进行填充
            if pad_length > 0:
                # 对 prompt 在序列末尾进行填充，填充 pad_token_id 表示空白 token
                batch["prompt"] = F.pad(prompt, pad=(0, pad_length), mode='constant', value=pad_token_id)
                # prompt_mask 在序列末尾进行填充，填充 0 表示对应的 attention mask 为无效
                batch["prompt_att_mask"] = F.pad(prompt_mask, pad=(0, pad_length),  mode='constant', value=0)
            # 因为上面已经在根据 max_prompt_seq_len 进行了过滤，所以这里肯定是 pad_length==0 的情况
            else:
                batch["prompt"] = prompt
                batch["prompt_att_mask"] = prompt_mask
            # 沿序列维度（也就是第 1 维）将 prompt  序列值反转回正常的。 最终其实就是在开头填充了 pad ，变成 [pad,pad,pad,1,2,3] 
            batch["prompt"] = batch["prompt"].flip(1)
             # 沿序列维度（也就是第 1 维）将  prompt_att_mask 序列值反转回正常的。最终其实就是在开头填充了 0 ，变成 [0,0,0,1,1,1]
            batch["prompt_att_mask"] = batch["prompt_att_mask"].flip(1)
            return batch  # （B，256）

# 经历数据制作

1.  `_generate_sequence`：这是一个辅助方法，用于根据 `prompts` 和 `mask` 生成 `actor_model` 模型的输出序列 seq ，并筛选过滤出答案长度大于 1 的有效的 seq 。简要步骤如下：

    1.  首先使用 `actor_model` 在`推理模式`下使用自回归方式模型针对 `prompt` 输出序列 `seq`，其中不仅包括 `prompt token` ，还有 `answer token` 、`padding token` 。
    2.  计算输出序列 `seq` 中每个 `batch` 维度的序列去掉 prompt 和 padding 之后的有效的答案序列长度，借用此结果滤掉 `seq` 中答案长度小于等于 1 的无效序列。
    3.  最后将经过过滤的有效序列 `seq` 按照 `batch` 维度再进行拼接成 `out_seq` 输出即可。

<!---->

    def _generate_sequence(self, prompts, mask):
        max_min_length = self.max_answer_seq_len + prompts.shape[1]  # max_answer_seq_len  是允许的最大答案长度，默认 256 ，提示词的长度已经统一成了 256 ，所以一共是 512 
        ...
        with torch.no_grad():
            seq = self.actor_model.module.generate( prompts, attention_mask=mask,  max_length=max_min_length, pad_token_id=self.tokenizer.pad_token_id, synced_gpus=self.z3_enabled, **kwargs)
        batch_size = seq.shape[0]
        prompt_length = prompts.shape[1]
        self.prompt_length = prompt_length
        # generate 方法会直接使用 prompts 的 input_ids 作为序列生成的起点。生成时会将 prompts 的内容保留，并在其后生成新的 token 。所以前面的 prompt_length 个 token 一定是和输入一样
        ans = seq[:, prompt_length:]
        # 计算序列中去掉 prompt 和 pad 之外的有效答案的长度
        valid_ans_len = (ans != self.tokenizer.pad_token_id).sum(dim=-1)
        ...
        out_seq = []
        for i in range(batch_size):
            # 如果有效答案的长度小于等于 1 被丢弃
            if valid_ans_len[i] <= 1:    
                continue
            else:
                out_seq.append(seq[i:i + 1])
        ...
        out_seq = torch.cat(out_seq, dim=0)   
        return out_seq  # （B，512）

2.  `generate_experience`：这个方法用于生成模型的经验历史数据，包括序列、对数概率、价值估计和奖励分数 等。这些数据将用于训练和优化模型。简要步骤如下：

    1.  `generate_experience` 方法首先在`推理模式`下使用自回归方式调用 `_generate_sequence` 方法生成 `actor_model` 模型生成的结果 seq ， seq 的每个 token 序列中包括 `prompt`、`answer` 和 `padding` 。
    2.  创建注意力掩码，以忽略序列中的 `pad` 填充标记。
    3.  在`训练模式`下使用四个模型 `actor_model 、ref_model、reward_model、critic_model` 处理答案序列，分别生成`对数概率、参考对数概率、奖励分数、价值估计`。
    4.  返回包含提示、对数概率、参考对数概率、价值、奖励、输入 ID 和注意力掩码的字典。

<!---->

    def generate_experience(self, prompts, mask):
        self.eval()
        ...
        seq = self._generate_sequence(prompts, mask)  # 见下面对 _generate_sequence 的解析 。seq 中包含了 prompt token、 answer token 、padding token ，(B,  max_prompt_seq_len +max_answer_seq_len 默认512)
        ...
        self.train()
        pad_token_id = self.tokenizer.pad_token_id
        attention_mask = seq.not_equal(pad_token_id).long()
        with torch.no_grad():
            # seq 只有生成的结果序列，而训练模式下使用模型进行解码会得到 【B，L，V】 的结果中，可以拿到每个位置的所有单词的 logits 
            output = self.actor_model(seq, attention_mask=attention_mask)
            output_ref = self.ref_model(seq, attention_mask=attention_mask)
            # 奖励模型 reward_model 对于 prompt+answer 的打分，取 values 中每个序列最后一个有效值作为奖励值，维度为(B,) , 通常用于强化学习的目标
            reward_score = self.reward_model.forward_value(seq, attention_mask, prompt_length=self.prompt_length)['chosen_end_scores'].detach()
            # 评论模型 critic_model 返回结果维度是(B,L)，L 维度上第 i 个位置代表从 i 位置到最后的累积奖励，用于辅助评估策略的好坏，舍去最后一个位置的 token
            # 价值函数  V(t) 是基于当前状态评估未来的期望回报，但最后一个 token 通常没有后续的未来信息，因此它的价值估计没有意义。
            # 而且生成任务是自回归的，序列的最后一个 token 不会为后续步骤提供任何预测依据，因为生成已经结束。
            values = self.critic_model.forward_value(seq, attention_mask, return_value_only=True).detach()[:, :-1]
        logits = output.logits
        logits_ref = output_ref.logits
        ...
        # logprobs 和  ref_logprobs 维度都是 (B, L)，L 上面每个位置就是 token 对应的对数概率值。
        # 每个时间步的 logit 对应当前时间步预测下一个 token 的分布。所以只有 logits[:, :-1] 的输出概率分布有效，正好针对 seq [:, 1:] 的 token
        # 简单例子 logits 【0.1, 1.2, 3.2】 
        #         seq   【我，   是，  谁】。其中  0.1 是对 “是”的预测概率分布，1.2 是对“谁”的预测概率分布，3.2 则是无效的
        return {'prompts': prompts,   # (B, max_prompt_seq_len 默认 256) 
                'input_ids': seq,   # (B, max_prompt_seq_len + max_answer_seq_len 默认512 )
                "attention_mask": attention_mask  # (B, max_prompt_seq_len + max_answer_seq_len 默认512 )
                'logprobs': gather_log_probs(logits[:, :-1, :], seq[:, 1:]),   # (B, L-1 默认 L 是 512)
                'ref_logprobs': gather_log_probs(logits_ref[:, :-1, :], seq[:, 1:]),   # (B, L-1 默认 L 是 512)
                'value': values,   # (B, L-1 默认 L 是 512)
                'rewards': reward_score,   # (B,) 
                }   

# 训练

1.  上面已经将经历数据制作完成，现在正式开始进行训练。将上面得到的一部分结果传入 `compute_rewards` 中用于计算奖励。上面我们制作数据的时候，是在**训练模式**下，将 `prompt+answer` 分别输入到 `actor_model` 和 `ref_model` ，得到的结果转为 `logprobs` 和 `ref_logprobs` 传入 `compute_rewards` 中用 `KL散度` 来衡量 `actor_model` 和 `ref_model` 输出的差别，也就是数据分布差距大小。同时将 `KL散度` 纳入 `损失函数` （KL散度本质是纳入到奖励值里边的，奖励值被纳入到了损失函数），进而来约束 `actor_model` 和 `ref_model` 的输出分布别差距太大。

<!---->

    def compute_rewards(self, prompts, log_probs, ref_log_probs, reward_score, action_mask):
        # KL散度的期望， log_probs 经过 log 变化，因此减法就对应除法，计算每个对应位置的 token 的对数概率差异
        # 计算了目标模型和参考模型之间的差异，并且通过 self.kl_ctl 调整这个差异的权重，加负号是因为要是最小化这个 KL 散度期望
        kl_divergence_estimate = -self.kl_ctl * (log_probs - ref_log_probs)
        rewards = kl_divergence_estimate
        # 只考虑 answer 部分的奖励，不考虑prompt
        start = prompts.shape[1] - 1
        ends = start + action_mask[:, start:].sum(1) + 1
        reward_clip = torch.clamp(reward_score, -self.clip_reward_value, self.clip_reward_value)
        batch_size = log_probs.shape[0]
        # 在L维度上，答案部分每个位置都有KL散度，但是只在最后一个位置加上奖励值
        for j in range(batch_size):
            rewards[j, start:ends[j]][-1] += reward_clip[j]
        return rewards  # (B, S)

2.  得到 compute\_rewards 计算的 old\_rewards 之后，还要将 old\_rewards 和 old\_values 答案有效位置之后的所有值都设置为 0 ，否则后面在计算 advantage/return 会出错。

<!---->

    old_values = inputs['value']
    with torch.no_grad():
        old_rewards = self.compute_rewards(prompts, log_probs, ref_log_probs, reward_score, action_mask)
        ends = start + action_mask[:, start:].sum(1) + 1 
        for i in range(old_rewards.shape[0]):
            old_rewards[i, ends[i]:] = 0
            old_values[i, ends[i]:] = 0
        advantages, returns = self.get_advantages_and_returns(old_values, old_rewards, start)

这里解释一下为什么会使用**优势函数**，其实收集一条完整轨迹并计算折扣回报，来进行后续的计算也是可以的，这种方法的优点是**无偏的**。因为我们不估计回报，我们使用的是获得的真实回报。但问题是**方差很大**，因为由于环境的随机性和策略的随机性，相同的起始状态可能导致非常不同的回报。**解决方案是使用大量轨迹数据** ，希望任何一条轨迹引入的方差总体上都会减少，并提供对回报的“真实”估计。然而，增加批次大小会显著**降低样本效率**。所以我们需要找到额外的机制来减少方差，也就是**优势函数**。

有了上面的 `rewards` 和 `values` ，我们就可以算 PPO 需要的优势值和回报值，有不懂的同学可以先学[这个概念及公式](https://huggingface.co/blog/deep-rl-a2c)，这里我们需要的是 Q 和 V 两个值函数，我们可以使用 `TD error` 来估计优势函数，这里的 `V` 对应的就是 `critic_model` 的输出，表示给定状态 s 下的平均值，Q 表示给定状态 s 下执行动作 a 的值。

![](https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/3e2494dfb5314724b144bd33e3fdec57~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAg5oiR5piv546L5aSn5L2g5piv6LCB:q75.awebp?policy=eyJ2bSI6MywidWlkIjoiNTM2MjE3NDA1ODk1MTQ5In0%3D&rk3s=f64ab15b&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1733205911&x-orig-sign=OEHUrk3Yapz1T%2BVSWzUrY7utQak%3D)

![](https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/3e74af1ce12c4d169773eed638868300~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAg5oiR5piv546L5aSn5L2g5piv6LCB:q75.awebp?policy=eyJ2bSI6MywidWlkIjoiNTM2MjE3NDA1ODk1MTQ5In0%3D&rk3s=f64ab15b&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1733205911&x-orig-sign=p7MquKvHfDoyVHr8lqyNJrbNvQc%3D)

    def get_advantages_and_returns(self, values, rewards, start):
        # values（B，L） critic_model 输出，包含每个 token 上的评分
        # rewards（B，L）reward_model 输出包含了kl散度以及最后一个有效答案 token 的奖励值
        # start 是 answer 开始的位置
        lastgaelam = 0
        advantages_reversed = []
        length = rewards.size()[-1]
        # 计算每个时刻（序列位置）的 critic_model 预测误差，注意这里是倒序计算的
        for t in reversed(range(start, length)):
            nextvalues = values[:, t + 1] if t < length - 1 else 0.0
            # critic_model 预测的是t到到最后一个时刻的奖励和，所以变化量delta可以用如下公式表示
            delta = rewards[:, t] + self.gamma * nextvalues - values[:, t]
            # self.gamma=1，self.lam=0.95是衰减因子，表示之前计算的delta对现在影响越来越小，衡量某个动作相对于基准的好坏程度，使用 GAE 平滑计算
            # 这种设计能够使优化既关注当前的即时奖励，又能兼顾未来的长期收益，从而提升整体性能。降低可能因为单步的随机奖励导致估计偏差较大的风险
            lastgaelam = delta + self.gamma * self.lam * lastgaelam
            advantages_reversed.append(lastgaelam)
        # 将结果进行反序，也就是扭成正常从左到右的顺序，再进行 stack 组合
        advantages = torch.stack(advantages_reversed[::-1], dim=1)
        # 优势加 values 中有效答案开始后的价值估计得到回报 returns ，后续用来更新 critic_model 
        returns = advantages + values[:, start:]
        return advantages.detach(), returns   # (B, start:length)

示例计算流程，假设：

*   `values = [[1.0, 1.2, 1.5]]`
*   `rewards = [[0.5, 0.7, 1.0]]`
*   `self.gamma = 0.99`
*   `self.lam = 0.95`

![](https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/4e99ebc929684a00a9397a9bc350ffd5~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAg5oiR5piv546L5aSn5L2g5piv6LCB:q75.awebp?policy=eyJ2bSI6MywidWlkIjoiNTM2MjE3NDA1ODk1MTQ5In0%3D&rk3s=f64ab15b&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1733205911&x-orig-sign=uHtb6VMsCYDLe6QlhhHcrXjNixM%3D)

3.  actor\_model 策略梯度损失，有了上面的所有值，我们就可以使用 PPO 算法计算 `actor_model` 的损失，我们可以对照下图的李宏毅老师的 公式 PPT 看代码。最后算出来的其实模型的策略梯度的损失。

![](https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/fa6f8d68592542658a21c3856ad96e13~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAg5oiR5piv546L5aSn5L2g5piv6LCB:q75.awebp?policy=eyJ2bSI6MywidWlkIjoiNTM2MjE3NDA1ODk1MTQ5In0%3D&rk3s=f64ab15b&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1733205911&x-orig-sign=XYmryEQEza3BYofnTpNQpuRTqbo%3D)

原始论文算法如下图：

![](https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/061774326c0a48bc874657d8a812007a~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAg5oiR5piv546L5aSn5L2g5piv6LCB:q75.awebp?policy=eyJ2bSI6MywidWlkIjoiNTM2MjE3NDA1ODk1MTQ5In0%3D&rk3s=f64ab15b&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1733205911&x-orig-sign=kEQSHdjMOvxwNju9yKgTdG2plf4%3D)

因为每批数据制作而成的经历数据会被 ppo 算法使用多次，虽然 inputs 部分的经历数据不会变，但是 actor 会更新多次，所以还要根据 inputs 中的内容处理出新的结果 actor\_log\_prob ，然后在借用上面计算出来的 advantages 来更新 actor 。

    batch = {'input_ids': seq, "attention_mask": attention_mask}
    actor_prob = self.actor_model(**batch, use_cache=False).logits
    actor_log_prob = gather_log_probs(actor_prob[:, :-1, :], seq[:, 1:])
    actor_loss = self.actor_loss_fn(actor_log_prob[:, start:],  # 最新的答案部分的模型输出概率分布
                                     log_probs[:, start:],    # 旧的答案部分的模型输出概率分布
                                     advantages,    # 固定的优势值
                                     action_mask[:, start:])  # 答案的 mask
    ...
    self.actor_model.backward(actor_loss)  # 计算梯度
    ...
    self.actor_model.step() # 更新模型
                                    
    def actor_loss_fn(self, logprobs, old_logprobs, advantages, mask):
        # 重要性采样权重计算 ratio = exp(log(new)-log(old))   都是经过log变化的单词概率，这里相当于在做新旧模型输出的每个 token 概率的除法
        log_ratio = (logprobs - old_logprobs) * mask
        ratio = torch.exp(log_ratio)
        # 计算 加权优势 与 裁剪加权优势
        pg_loss1 = -advantages * ratio
        pg_loss2 = -advantages * torch.clamp(ratio, 1.0 - self.cliprange, 1.0 + self.cliprange)
        # 公式里面是最小值，我猜测这里使用最大值 max 这里因为入参是负数
        # 从2种情况中选择损失较大者作为真正的损失， 并且基于 batch 内所有数据的所有有效时间步计算平均损失值
        pg_loss = torch.sum(torch.max(pg_loss1, pg_loss2) * mask) / mask.sum()
        return pg_loss

5.  critic\_model 损失，其实就 MSE 算法计算出来的损失。和上面同样的道理，critic 会更新多次 我们要要根 inputs 中的内容处理出新的结果 value ，再借用上面计算出来的 returns 更新 critic 模型。

<!---->

    value = self.critic_model.forward_value(**batch, return_value_only=True, use_cache=False)[:, :-1]
    critic_loss = self.critic_loss_fn(value[:, start:], old_values[:,  start:],  returns, action_mask[:, start:])
    self.critic_model.backward(critic_loss)
    ...
    self.critic_model.step()

    def critic_loss_fn(self, values, old_values, returns, mask):
        ## value loss 需要注意的是这里使用裁剪的“老critic_model”的输出约束“新critic_model”不要步子太大。
        values_clipped = torch.clamp(values, old_values - self.cliprange_value,old_values + self.cliprange_value,)
        ...
        vf_loss1 = (values - returns)**2
        vf_loss2 = (values_clipped - returns)**2
        vf_loss = 0.5 * torch.sum(torch.max(vf_loss1, vf_loss2) * mask) / mask.sum()
        return vf_loss

6.  使用经历数据多次更新 ppo 算法全过程

每个 batch 的 prompt 数据，都会被制作成经历数据重复使用多次来更新 ppo 算法，具体细节就是上面的介绍过的每一部分：

    def train_rlhf(self, inputs):
        prompts = inputs['prompts']
        log_probs = inputs['logprobs']
        ref_log_probs = inputs['ref_logprobs']
        reward_score = inputs['rewards']
        values = inputs['value']
        attention_mask = inputs['attention_mask']
        seq = inputs['input_ids']

        start = prompts.size()[-1] - 1
        action_mask = attention_mask[:, 1:]
        old_values = values
        # 计算优势和回报值
        with torch.no_grad():
            old_rewards = self.compute_rewards(prompts, log_probs, ref_log_probs, reward_score, action_mask)
            ends = start + action_mask[:, start:].sum(1) + 1 
            for i in range(old_rewards.shape[0]):
                old_rewards[i, ends[i]:] = 0
                old_values[i, ends[i]:] = 0
            advantages, returns = self.get_advantages_and_returns(old_values, old_rewards, start)

        # 更新 actor_model
        batch = {'input_ids': seq, "attention_mask": attention_mask}
        actor_prob = self.actor_model(**batch, use_cache=False).logits
        actor_log_prob = gather_log_probs(actor_prob[:, :-1, :], seq[:, 1:])
        actor_loss = self.actor_loss_fn(actor_log_prob[:, start:], log_probs[:, start:], advantages,  action_mask[:, start:])
        self.actor_model.backward(actor_loss)
        ...
        self.actor_model.step()
        ...
        
        # 更新 critic_model
        value = self.critic_model.forward_value(**batch, return_value_only=True,  use_cache=False)[:, :-1]
        critic_loss = self.critic_loss_fn(value[:, start:], old_values[:, start:], returns, action_mask[:, start:])
        self.critic_model.backward(critic_loss)
        ...
        self.critic_model.step()
        ...
        
        return actor_loss, critic_loss

到此整个 `RLHF` 的核心逻辑算是基本介绍完了。

# 参考

*   <https://zhuanlan.zhihu.com/p/624589622>
*   <https://huggingface.co/blog/rlhf>
*   <https://mathmach.com/be7f3b4f/>
*   <https://github.com/microsoft/DeepSpeedExamples/blob/master/applications/DeepSpeed-Chat/dschat/utils/model/reward_model.py>
