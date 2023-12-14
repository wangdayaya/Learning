# 前言

本文使用 DDPG（Deep Deterministic Policy Gradient） 强化学习算法玩 Pendulum 游戏。

#  对比 DDPG 和 DQN 
[前文](https://juejin.cn/post/7311343196628959259)我们已经介绍过 Deep Q-leanrning 算法，它和 DDPG 是两种不同类型的强化学习算法，它们在目标、适用场景和基本原理等方面有一些显著的区别。

1. **学习目标：**
   - **DDPG：** 主要用于解决连续动作空间的问题，其目标是学习一个确定性策略，能够映射状态到连续的动作空间。
   - **DQN：** 用于解决离散动作空间的问题，其目标是学习一个值函数，估计每个状态-动作对的累积奖励。

2. **动作空间：**
   - **DDPG：** 适用于连续动作空间，因为它的输出是一个确定性的动作值。
   - **DQN：** 主要用于离散动作空间，因为它需要为每个动作输出一个 Q 值。

3. **算法基础：**
   - **DDPG：** 是一种基于策略梯度方法的算法，直接学习一个确定性的策略。
   - **DQN：** 是一种基于值函数的算法，通过学习Q值函数来选择最优的动作。

4. **经验回放：**
   - **DDPG：** 借用了经验回放（experience replay）来存储和重新使用先前的经验，提高样本的利用效率和算法的稳定性。
   - **DQN：** 也使用经验回放，它通过从过去的经验中随机抽样来训练神经网络，增强样本的独立性，提高算法的稳定性。

5. **算法架构：**
   - **DDPG：** 使用Actor-Critic架构，包括一个策略网络（Actor）和一个值函数网络（Critic）。
   - **DQN：** 使用单一的深度神经网络来估计 Q 值。

6. **目标网络：**
   - **DDPG：** 除了本身的 actor 和 critic 模型，还引入了目标 actor 和目标 critic ，通过定期更新它们的参数，以稳定训练过程。
   - **DQN：** 也使用了 target 模型，通过定期更新目标网络的参数来提高算法的稳定性。

# 游戏介绍


经典的 Pendulum 控制问题。 这个问题很难使用 Q-Learning 算法来解决，因为动作是连续的而不是离散的，也就是说，我们必须从 -2 到 +2 的无限动作中选择一个实数作为力矩来控制小棒的摆动，尽量保证小棒尽量一直保持竖直向上，这样可以持续得分，最后得分越高越好。


![pendulum .gif](https://p6-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/d950fcdeee1340c1b561ce2d62644f72~tplv-k3u1fbpfcp-jj-mark:0:0:0:0:q75.image#?w=456&h=326&s=582949&e=gif&f=74&b=ffffff)


1. **动作空间** 

Pendulum 的动作空间属于连续动作空间，动作空间大小 1

-   `控制力矩` : 大小范围是 `[-2, 2]` 。


2. **状态空间** 

Pendulum 的状态空间大小 3，描述了钟摆的角度和角速度。分别是：

-   `sin` ：钟摆偏离竖直方向角度的 sin 值，范围是 `[-1, 1]` 。
-   `cos` ：钟摆偏离竖直方向角度的 cos 值，范围是 `[-1, 1]` 。
-   `thetadot` ：钟摆的角角度，范围是 `[-8, 8]` 。


# 模型介绍


下面代码是 actor 模型的结构，主要是简单的非线性变化神经网络结构，需要注意的是我们需要 `Actor` 最后一层的激活函数使用 `tanh` 会使得输出的 action 动作在 `1 或 -1 之间`，但是因为我们最后还要对模型输出结果缩放到 `-2 到 2 之间`，这可能会使梯度近乎减小为 0 ，不利于求导和训练，所以我们使用函数 `random_uniform_initializer` 初始化权重介于 `-0.003 和 0.003` 之间。

```
def get_actor():
    last_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)
    inputs = layers.Input(shape=(num_states,))
    out = layers.Dense(256, activation="relu")(inputs)
    out = layers.Dense(256, activation="relu")(out)
    outputs = layers.Dense(1, activation="tanh", kernel_initializer=last_init)(out)
    outputs = outputs * upper_bound
    model = tf.keras.Model(inputs, outputs)
    return model
```



下面是 critic 模型的机构，神经网络结构也不复杂。需要注意的是，为什么 `Critic 模型`中要接受 `state 和 action 两个输入`，而不是像 Deep Q-learning中只接受 state 一个输入呢？在连续动作空间中，动作可以是任意的实数值，而不是离散的动作空间。这使得在计算 Q 值时面临一些挑战，因为无法简单地为每一个可能的动作都计算一个具体的 Q 值。Critic 模型的目标是估计`状态-动作对`的 Q 值，这个值表示`在当前状态下采取某个动作的长期奖励预期是多少`。这种设计允许 Critic 模型更好地理解状态和动作之间的关系，从而更准确地指导 Actor 生成策略，提高在连续动作空间中的强化学习性能。

```
def get_critic():
    state_input = layers.Input(shape=num_states)
    state_out = layers.Dense(16, activation="relu")(state_input)
    state_out = layers.Dense(32, activation="relu")(state_out)
    action_input = layers.Input(shape=num_actions)
    actino_out = layers.Dense(32, activation="relu")(action_input)
    concat = layers.Concatenate()([state_out, actino_out])
    out = layers.Dense(256, activation="relu")(concat)
    out = layers.Dense(256, activation="relu")(out)
    outputs = layers.Dense(1)(out)
    model = tf.keras.Model([state_input, action_input], outputs)
    return model
```



下面的代码中展示了，基于状态和随机噪声进行动作采样的过程。其中的 noise_object 实现了一个Ornstein-Uhlenbeck（OU）过程的动作噪声生成器，通常用于强化学习中的连续动作空间问题。OU 过程是一种随机过程，用于模拟具有一定持续性的随机变化。在强化学习中，这种噪声通常被添加到连续动作中，以增加探索性和稳定性。

```
def policy(state, noise_object):
    sampled_actions = tf.squeeze(actor_model(state))
    noise = noise_object()
    sampled_actions = sampled_actions.numpy() + noise
    legal_action = np.clip(sampled_actions, lower_bound, upper_bound)
    return [np.squeeze(legal_action)]
```

# 模型训练

下图是 DDPG 模型的主要思想，损失计算和模型梯度更新实现的关键代码如下：

<p align=center><img src="https://p6-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/71b389486c894afe8fb1dfba11ffb989~tplv-k3u1fbpfcp-jj-mark:0:0:0:0:q75.image#?w=553&h=401&s=189075&e=png&b=fdfbfb" alt="image.png"  /></p>


需要注意的是，我们不仅要训练 critic 和 actor ，另外还要定期更新 target_critic 和 target_actor 。
```
with tf.GradientTape() as tape:
    target_actions = target_actor(next_state_batch, training=True)
    y = reward_batch + gamma * target_critic([next_state_batch, target_actions], training=True)
    critic_value = critic_model([state_batch, action_batch], training=True)
    critic_loss = tf.math.reduce_mean(tf.math.square(y - critic_value))
critic_grad = tape.gradient(critic_loss, critic_model.trainable_variables)
critic_optimizer.apply_gradients(zip(critic_grad, critic_model.trainable_variables))

with tf.GradientTape() as tape:
    actions = actor_model(state_batch, training=True)
    critic_value = critic_model([state_batch, actions], training=True)
    actor_loss = -tf.math.reduce_mean(critic_value)
actor_grad = tape.gradient(actor_loss, actor_model.trainable_variables)
actor_optimizer.apply_gradients(zip(actor_grad, actor_model.trainable_variables))

def update_target(target_weights, weights, tau):
    for (a, b) in zip(target_weights, weights):
        a.assign(b * tau + a * (1 - tau))
```

训练日志打印：

    Episode * 0 * Avg Reward is ==> -1157.8143517891501
    Episode * 1 * Avg Reward is ==> -1189.9761222732382
    Episode * 2 * Avg Reward is ==> -1044.89974715879
    ...
    Episode * 36 * Avg Reward is ==> -557.5050289524168
    Episode * 37 * Avg Reward is ==> -546.1769189528364
    Episode * 38 * Avg Reward is ==> -535.4149222739205
    ...
    Episode * 97 * Avg Reward is ==> -183.08817536189082
    Episode * 98 * Avg Reward is ==> -183.19096335666998
    Episode * 99 * Avg Reward is ==> -183.10935571438955

100 轮游戏得分趋势图如下，可以看出强化学习能有效提升游戏的得分：

<p align=center><img src="https://p3-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/d5845b2da5574e588f6af69b2670406a~tplv-k3u1fbpfcp-jj-mark:0:0:0:0:q75.image#?w=640&h=480&s=19587&e=png&b=ffffff" alt="image.png"  /></p>

# 效果

在初始状态下，小棒始终都是随机摆动，无法保持垂直竖起来的状态。

<p align=center><img src="https://p1-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/492b9b4366314a1381d77d6cc6d56e61~tplv-k3u1fbpfcp-jj-mark:0:0:0:0:q75.image#?w=500&h=500&s=286065&e=gif&f=67&b=ffffff" alt="pendulum初期 .gif"  /></p>

经过强化训练之后的小棒能够在采取的不同力矩动作下，保持长时间的竖直状态，能够获得较高的分数。

<p align=center><img src="https://p9-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/0a87a74da4ff4a168a4ce5f906fd3567~tplv-k3u1fbpfcp-jj-mark:0:0:0:0:q75.image#?w=500&h=500&s=208346&e=gif&f=40&b=ffffff" alt="pendulum 训练.gif"  /></p>

# 参考

