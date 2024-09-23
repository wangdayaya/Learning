# 前文
我对 RLHF 的具体细节一直模棱两可，所以干脆找了微软的项目 [DeepSpeed-Chat](https://github.com/microsoft/DeepSpeedExamples/tree/master/applications/DeepSpeed-Chat) 中的具体实现的代码拿来研究细节，`我参考了几位大佬的文章，尽量用自己最简洁的语言介绍清楚，我这里只抓核心代码，无关紧要的代码都被我删掉了，不影响整体逻辑`。



# 准备

本文只讲 PPO 具体代码细节，我这里不会详细解释大语言模型基础和强化学习的入门概念，所以需要您自行学习准备，如：

- Transformer
- LLM
- RLHF
- 动作价值函数
- 状态价值函数
- 最优动作价值函数
- 最优状态价值函数
- 折扣回报
- 奖励
- 动作
- 状态
- 策略
- 优势函数
- PPO 算法
- ...

# SFT

第一步 `SFT` 过程默认大家都会，这里只介绍`第二阶段`和 `第三阶段`两部分。

# 奖励模型

## 架构图

![image.png](https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/5868c10aae7c431f92ffdf0a45e369a8~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAg5oiR5piv546L5aSn5L2g5piv6LCB:q75.awebp?policy=eyJ2bSI6MywidWlkIjoiNTM2MjE3NDA1ODk1MTQ5In0%3D&rk3s=e9ecf3d6&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1727146498&x-orig-sign=NnXC2jQOpQBX%2BO0iAs%2BIDDM3hbw%3D)

我们可以回看源代码中的[奖励模型](https://github.com/microsoft/DeepSpeedExamples/blob/master/applications/DeepSpeed-Chat/dschat/utils/model/reward_model.py)的具体实现，这里主要介绍它的输出和训练过程。需要注意的是这份奖励模型代码后面在 ` PPO 算法优化阶段 `被使用到，创建出两个一样的副本模型 `reward_model 和 critic_model` ，只是它们的所输出的值有所不同。

## 输出
奖励模型的输出可以查看奖励模型源代码中的 ```forward_value ```函数，结果输出主要有两部分组成，代码中是这样写的：

```
return {
    "values": values,  # 【B，L】
    "chosen_end_scores": torch.stack(chosen_end_scores),  #【B，】
}
```
- `values` 中某个 `batch` 结果的第 `i` 个位置的值表示：从第 `i` 个位置到最后一个位置输出所能获得的奖励分值的累加和（和 `DQN` 里边的 `Q` 值一个意义， Q(s, a) 表示在给定状态 ( s ) 下，采取动作 ( a ) 并遵循某策略后，智能体从当前状态到未来状态所能获得的预期总回报），这个输出满足了后续 `PPO` 算法优化阶段 `critic_model` 的输出要求。维度为 `(B,L)` 。
- `chosen_end_scores` 表示的是对于 `prompt` ， `SFT` 做出的的 `answer` 的评分，这个输出满足了后续 `PPO` 算法优化阶段 `reward_model` 的输出要求。其实 `chosen_end_scores` 也是通过将 `values` 经过处理拿到的。`chosen_end_scores` 列表包含了每个模型输出答案的评分，这些分数是基于序列的最后一个有效标记（不是填充标记）的值，这个值可以用于评估模型生成的答案质量，维度是 `(B,)` 。


## 训练

模型训练优化采用 `pair wiss loss`，即输入模型关于同一个问题的两个回答，让模型学会这两个句子哪个分高哪个分低。代码如下：

```
# 同一个 batch 里边的句子需要等长，所以会被 padding 
# [divergence_ind:end_ind] 索引了 padding 前一个位置的输出分值
# chosen_reward 是 pair 里排序靠前的答案的分数，r_truncated_reward 是 pair 里排序靠后的答案的分数
# 在这里的两个分数，其实和上面你的 chosen_end_scores 在本质上和实现上是一个东西
c_truncated_reward = chosen_reward[divergence_ind:end_ind]
r_truncated_reward = rejected_reward[divergence_ind:end_ind]
```

`pair wise loss` 代码如下，目标就是尽量给 `pair` 里边好的答案打分高（c_truncated_reward），给差的答案（r_truncated_reward）打分低，这样 loss 才能变小：

```
loss += -torch.nn.functional.logsigmoid(c_truncated_reward - r_truncated_reward).mean()
```

经过训练，奖励模型的输入是 `prompt+answer` 的形式，让模型学会对 `prompt+answer` 进行打分即可，最后返回的是一个 `scalar` 。

# PPO 算法优化

下面介绍的代码中会将不重要的代码删减，不影响算法逻辑，具体源代码实现可以回看 [ppo_trainer.py](https://github.com/microsoft/DeepSpeedExamples/blob/master/applications/DeepSpeed-Chat/dschat/rlhf/ppo_trainer.py) 、[reward_model.py](https://github.com/microsoft/DeepSpeedExamples/blob/master/applications/DeepSpeed-Chat/dschat/utils/model/reward_model.py)、[main.py](https://github.com/microsoft/DeepSpeedExamples/blob/master/applications/DeepSpeed-Chat/training/step3_rlhf_finetuning/main.py)。

## 架构图

整个 `PPO` 算法，涉及到`四个模型`，其中 `actor_model 和 ref_model` 一开始初始化都是第一阶段 `SFT` 模型的相同副本，只不过下面只更新 `actor_model`，但是强化学习过程很容易把模型训练“坏”，因此需要另外一个**参数不会更新或者某种策略慢更新**（如：直接 copy 或者 EMA 等策略）的 `ref_model` 来当作标的，别让 `actor mode` 跑偏太远。 `reward_model 和 critic_model` 一开始初始化都是第二阶段奖励模型的相同副本，只不过下面只更新 `critic_model` 。在这个图中没有画出来 `critic model` ，但是实际上在代码实现中是存在的。
 
![image.png](https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/ac69db4a791848b991d379f46a947cf1~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAg5oiR5piv546L5aSn5L2g5piv6LCB:q75.awebp?policy=eyJ2bSI6MywidWlkIjoiNTM2MjE3NDA1ODk1MTQ5In0%3D&rk3s=e9ecf3d6&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1726907592&x-orig-sign=EPVgzK%2FCBkJYZGYGHDMUXBd2Mqw%3D)


## 训练模式和推理模式

`LLM` 的训练模式使用 `teacher force` 的方式，将整句话输入到模型中，并通过 `mask` 机制在保证不泄漏未来的单词情况下预测下一个单词。推理模式是真正的自回归，预测出下一个单词之后，当作下一步输入再预测下下一个单词。


![image.png](https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/92cb5a055829446cb09ffa129479c0d9~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAg5oiR5piv546L5aSn5L2g5piv6LCB:q75.awebp?policy=eyJ2bSI6MywidWlkIjoiNTM2MjE3NDA1ODk1MTQ5In0%3D&rk3s=e9ecf3d6&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1727145881&x-orig-sign=pu6VuNs7b3oXJ9SiwtLpYztdb8M%3D)

## PPO 算法需要的输入

1. `generate_experience`：这个方法用于生成模型的经验历史数据，包括序列、对数概率、价值估计和奖励分数。这些数据将用于训练和优化模型。简要步骤如下：

- `generate_experience` 方法首先在`推理模式`下使用自回归方式调用 `_generate_sequence` 方法生成 `actor_model` 模型生成的结果，其中不仅包括 `prompt` ，还有答案。
- 创建注意力掩码，以忽略序列中的 `pad` 填充标记。
- 在`训练模式`下使用四个模型 `actor_model 、ref_model、reward_model、critic_model` 处理答案序列，分别生成`对数概率、参考对数概率、奖励分数、价值估计`。
- 返回包含提示、对数概率、参考对数概率、价值、奖励、输入 ID 和注意力掩码的字典。

2. `_generate_sequence`：这是一个辅助方法，用于生成 `actor_model` 模型的输出序列。简要步骤如下：

- 首先使用 `actor_model` 在`推理模式`下使用自回归方式模型针对 `prompt` 输出序列  `seq`，其中不仅包括 `prompt` ，还有答案。
- 将输出序列  `seq` 中每个 `batch` 维度的序列中的 `prompt` 部分去掉，只剩下有效的答案序列，借用此结果滤掉  `seq` 中答案长度小于等于 1 的无效序列。
- 最后将经过过滤的有效序列 `seq` 按照 `batch` 维度进行拼接成 `out_seq` 输出即可。

```
def generate_experience(self, prompts, mask):
    self.eval()
    seq = self._generate_sequence(prompts, mask)
    self.train()
    pad_token_id = self.tokenizer.pad_token_id
    attention_mask = seq.not_equal(pad_token_id).long()
    with torch.no_grad():
        # 训练模式下使用模型进行解码会得到 【B，L，V】 的结果中可以拿到每个位置的所有单词的 logits 
        output = self.actor_model(seq, attention_mask=attention_mask)
        output_ref = self.ref_model(seq, attention_mask=attention_mask)
        # 奖励模型就是上面解释的 chosen_end_scores ，表示对于 prompt+answer 的打分，维度为(B,)
        reward_score = self.reward_model.forward_value(seq, attention_mask, prompt_length=self.prompt_length)['chosen_end_scores'].detach()
        # critic_model 返回结果维度是(B,L)，L 维度上第 i 个位置代表从 i 位置到最后的累积奖励
        # 舍去最后一个位置是因为句子“终止符”无意义 
        values = self.critic_model.forward_value(seq, attention_mask, return_value_only=True).detach()[:, :-1]
    logits = output.logits
    logits_ref = output_ref.logits
    # logprobs 和  ref_logprobs 维度都是 (B,L) ，L 上面每个位置就是 token 对应的对数概率值。
    return {'prompts': prompts, 'logprobs': gather_log_probs(logits[:, :-1, :], seq[:, 1:]), 'ref_logprobs': gather_log_probs(logits_ref[:, :-1, :], seq[:, 1:]), 'value': values, 'rewards': reward_score, 'input_ids': seq, "attention_mask": attention_mask }

def _generate_sequence(self, prompts, mask):
    max_min_length = self.max_answer_seq_len + prompts.shape[1]
    with torch.no_grad():
        seq = self.actor_model.module.generate( prompts, attention_mask=mask,  max_length=max_min_length, pad_token_id=self.tokenizer.pad_token_id, synced_gpus=self.z3_enabled, **kwargs)
    batch_size = seq.shape[0]
    prompt_length = prompts.shape[1]
    self.prompt_length = prompt_length
    ans = seq[:, prompt_length:]
    valid_ans_len = (ans != self.tokenizer.pad_token_id).sum(dim=-1)
    out_seq = []
    for i in range(batch_size):
        if valid_ans_len[i] <= 1:  # if the answer is shorter than 1 token, drop it ，NOTE: this will causes each GPU has different number of examples
            continue
        else:
            out_seq.append(seq[i:i + 1])
    out_seq = torch.cat(out_seq, dim=0)  # concat output in the batch dim
    return out_seq
```

将上面得到的部分结果传入 `compute_rewards` 中用于计算奖励，我们在**训练模式**下，将 prompt+answer 分别输入到 `actor_model` 和 `ref_model` ，用 `KL散度` 来衡量  `actor_model` 和 `ref_model` 输出的差别，也就是数据分布差距大小。同时将 `KL散度` 纳入 `损失函数` （KL散度本质是纳入到奖励值里边的，奖励值被纳入到了损失函数），进而来约束  `actor_model` 和 `ref_model` 的输出分布别差距太大。


```
def compute_rewards(self, prompts, log_probs, ref_log_probs, reward_score, action_mask):
    #  log_probs 经过 log 变化，因此减法就对应除法
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
    return rewards
```


有了上面的 `rewards` 和 `values` ，我们就可以算 PPO 需要的优势值和回报值，又不懂的同学可以先学些[这个概念及公式](https://huggingface.co/blog/deep-rl-a2c)，V 对应的就是 `critic_model` 的输出。

![image.png](https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/8ad9eb06123248859dd9730afac77750~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAg5oiR5piv546L5aSn5L2g5piv6LCB:q75.awebp?policy=eyJ2bSI6MywidWlkIjoiNTM2MjE3NDA1ODk1MTQ5In0%3D&rk3s=e9ecf3d6&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1727148632&x-orig-sign=%2B82MlMExNv9XH6vIdFBntmc2vkE%3D)


```
def get_advantages_and_returns(self, values, rewards, start):
    # values（B，L） critic_model 输出、 rewards（B，L）reward 包含kl散度、 start answer开始的位置
    lastgaelam = 0
    advantages_reversed = []
    length = rewards.size()[-1]
    # 计算每个时刻（序列位置）的 critic_model 预测误差，注意这里是倒序计算的
    for t in reversed(range(start, length)):
        nextvalues = values[:, t + 1] if t < length - 1 else 0.0
        # critic_model 预测的是t到到最后一个时刻的奖励和，所以变化量delta可以用如下公式表示
        delta = rewards[:, t] + self.gamma * nextvalues - values[:, t]
        # self.gamma=1，self.lam=0.95是衰减因子，表示之前计算的delta对现在影响越来越小
        lastgaelam = delta + self.gamma * self.lam * lastgaelam
        advantages_reversed.append(lastgaelam)
    # 将结果进行反序，也就是扭成正常从左到右的顺序，再进行 stack 组合
    advantages = torch.stack(advantages_reversed[::-1], dim=1)
    # 后续用来更新 critic_model 用
    returns = advantages + values[:, start:]
    return advantages.detach(), returns
```



## actor_model 策略梯度损失

有了上面的所有值，我们就可以使用 PPO 算法计算  `actor_model` 的损失，我们可以对照下图的李宏毅老师的 公式 PPT 看代码。最后算出来的其实模型的策略梯度的损失。

![image.png](https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/0bb3bd3113e447e4a9e0bd5dda3133f8~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAg5oiR5piv546L5aSn5L2g5piv6LCB:q75.awebp?policy=eyJ2bSI6MywidWlkIjoiNTM2MjE3NDA1ODk1MTQ5In0%3D&rk3s=e9ecf3d6&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1727159629&x-orig-sign=0TrfMFSm2HJHSAcUWahqqew1w4w%3D)
```
def actor_loss_fn(self, logprobs, old_logprobs, advantages, mask):
    # policy gradient loss， 都是经过log变化的单词概率，这里相当于在做概率除法
    log_ratio = (logprobs - old_logprobs) * mask
    ratio = torch.exp(log_ratio)
    pg_loss1 = -advantages * ratio
    pg_loss2 = -advantages * torch.clamp(ratio, 1.0 - self.cliprange, 1.0 + self.cliprange)
    # 公式里面是最小值，我猜测这里使用最大值 max 这里因为入参是负数
    pg_loss = torch.sum(torch.max(pg_loss1, pg_loss2) * mask) / mask.sum()
    return pg_loss
```

## critic_model 损失
其实就 MSE 算法计算出来的损失。需要注意的是这里使用裁剪的“老critic_model”的输出约束“新critic_model”不要步子太大。

```
def critic_loss_fn(self, values, old_values, returns, mask):
    ## value loss
    values_clipped = torch.clamp(values,old_values - self.cliprange_value,old_values + self.cliprange_value,)
    vf_loss1 = (values - returns)**2
    vf_loss2 = (values_clipped - returns)**2
    vf_loss = 0.5 * torch.sum(torch.max(vf_loss1, vf_loss2) * mask) / mask.sum()
    return vf_loss
```


## 模型更新训练

下面就是更新 `actor_model` 和 `critic_model` 的代码，这里需要特别注意的是：

- 在计算 `advantages` 过程中使用的是历史数据，没有使用最新的模型进行计算，因为计算它所需要的都是历史数据。
- 在计算 `get_advantages_and_returns` 的时候，需要将 `old_rewards` 和 `old_values` 中每个 batch 维度的有效答案之后的位置的值都设置为 0 ，否则会影响 `advantage/return` 的计算。
- 计算 `actor_loss_fn` 使用最新的 `actor_model` 对 `prompt` 计算的结果 `actor_log_prob` 当作了入参 `logprobs`，而使用了历史数据 `log_probs` 当作了入参 `old_logprobs` 。这是因为在训练过程中，模型的参数会不断更新。直接使用 `inputs` 中的旧数据可能会导致计算出的损失与当前模型状态不一致。通过重新计算，可以确保使用的是最新的模型参数。
- 计算 `critic_loss_fn` 使用最新的 `critic_model` 对 `prompt` 计算的结果 `value` 当作了入参 `values`，而使用了历史数据 `old_values` 当作了入参 `old_values` 。原因同上。


```
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
        # we need to zero out the reward and value after the end of the conversation
        # otherwise the advantage/return will be wrong
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
    self.actor_model.step()
    
    # 更新 critic_model
    value = self.critic_model.forward_value(**batch, return_value_only=True,  use_cache=False)[:, :-1]
    critic_loss = self.critic_loss_fn(value[:, start:], old_values[:, start:], returns, action_mask[:, start:])
    self.critic_model.backward(critic_loss)
    self.critic_model.step()

    return actor_loss, critic_loss
```

到此整个 `RLHF` 的核心逻辑算是基本介绍完了。


# 参考

- https://zhuanlan.zhihu.com/p/624589622
- https://huggingface.co/blog/rlhf
- https://mathmach.com/be7f3b4f/
- https://github.com/microsoft/DeepSpeedExamples/blob/master/applications/DeepSpeed-Chat/dschat/utils/model/reward_model.py