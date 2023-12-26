

## CartPole 介绍


在一个光滑的轨道上有个推车，杆子垂直微置在推车上，随时有倒的风险。系统每次对推车施加向左或者向右的力，但我们的目标是让杆子保持直立。杆子保持直立的每个时间单位都会获得 +1 的奖励。但是当杆子与垂直方向成 15 度以上的位置，或者推车偏离中心点超过 2.4 个单位后，这一轮局游戏结束。因此我们可以获得的最高回报等于 200 。我们这里就是要通过使用 PPO 算法来训练一个强化学习模型   actor-critic ，通过对比模型训练前后的游戏运行 gif 图，可以看出来我们训练好的模型能长时间保持杆子处于垂直状态。

# 库准备

    python==3.10.9
    tensorflow-gpu==2.10.0
    imageio==2.26.1
    keras==2.10,0
    gym==0.20.0
    pyglet==1.5.20
    scipy==1.10.1通过 CartPole 游戏详细说明 PPO 优化过程

## 超参数设置

这段代码主要是导入所需的库，并设置了一些超参数。

        import numpy as np
        import tensorflow as tf
        from tensorflow import keras
        from tensorflow.keras import layers
        import gym
        import scipy.signal
        import time
        from tqdm import tqdm

        steps_per_epoch = 5000  # 每个 epoch 中训练的步数
        epochs = 20  # 用于训练的 epoch 数
        gamma = 0.90  # 折扣因子，用于计算回报
        clip_ratio = 0.2  # PPO 算法中用于限制策略更新的比率
        policy_learning_rate = 3e-4  # 策略网络的学习率
        value_function_learning_rate = 3e-4  # 值函数网络的学习率
        train_policy_iterations = 80  # 策略网络的训练迭代次数
        train_value_iterations = 80  # 值函数网络的训练迭代次数
        lam = 0.97  # PPO 算法中的 λ 参数
        target_kl = 0.01  # PPO 算法中的目标 KL 散度
        hidden_sizes = (64, 64) # 神经网络的隐藏层维度 
        render = False    # 是否开启画面渲染，False 表示不开启
    
## 模型定义


（1）这里定义了一个函数 `discounted_cumulative_sums`，接受两个参数 `x` 和 `discount`，该函数的作用是计算给定奖励序列 `x` 的折扣累计和，折扣因子 `discount` 是一个介于 0 和 1 之间的值，表示对未来奖励的折扣程度。 在强化学习中，折扣累计和是一个常用的概念，表示对未来奖励的折扣累加。

    def discounted_cumulative_sums(x, discount):
        return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]

（2）这里定义了一个Buffer类，用于存储训练数据。类中有如下主要的函数：

-   init: 初始化函数，用于设置成员变量的初始值
-   store: 将观测值、行为、奖励、价值和对数概率存储到对应的缓冲区中
-   finish_trajectory: 结束一条轨迹，用于计算优势和回报，并更新 trajectory_start_index 的值
-   get: 获取所有缓冲区的值，用在训练模型过程中。在返回缓冲区的值之前，将优势缓冲区的值进行标准化处理，使其均值为 0 ，方差为 1 


        class Buffer:
            def __init__(self, observation_dimensions, size, gamma=0.99, lam=0.95):
                self.observation_buffer = np.zeros( (size, observation_dimensions), dtype=np.float32 )
                self.action_buffer = np.zeros(size, dtype=np.int32)
                self.advantage_buffer = np.zeros(size, dtype=np.float32)
                self.reward_buffer = np.zeros(size, dtype=np.float32)
                self.return_buffer = np.zeros(size, dtype=np.float32)
                self.value_buffer = np.zeros(size, dtype=np.float32)
                self.logprobability_buffer = np.zeros(size, dtype=np.float32)
                self.gamma, self.lam = gamma, lam
                self.pointer, self.trajectory_start_index = 0, 0

            def store(self, observation, action, reward, value, logprobability):
                self.observation_buffer[self.pointer] = observation
                self.action_buffer[self.pointer] = action
                self.reward_buffer[self.pointer] = reward
                self.value_buffer[self.pointer] = value
                self.logprobability_buffer[self.pointer] = logprobability
                self.pointer += 1

            def finish_trajectory(self, last_value=0):
                path_slice = slice(self.trajectory_start_index, self.pointer)
                rewards = np.append(self.reward_buffer[path_slice], last_value)
                values = np.append(self.value_buffer[path_slice], last_value)
                deltas = rewards[:-1] + self.gamma * values[1:] - values[:-1]
                self.advantage_buffer[path_slice] = discounted_cumulative_sums( deltas, self.gamma * self.lam )
                self.return_buffer[path_slice] = discounted_cumulative_sums(  rewards, self.gamma )[:-1]
                self.trajectory_start_index = self.pointer

            def get(self):
                self.pointer, self.trajectory_start_index = 0, 0
                advantage_mean, advantage_std = (  np.mean(self.advantage_buffer),  np.std(self.advantage_buffer), )
                self.advantage_buffer = (self.advantage_buffer - advantage_mean) / advantage_std
                return ( self.observation_buffer, self.action_buffer, self.advantage_buffer, self.return_buffer, self.logprobability_buffer, )



（3）这里定义了一个多层感知机（Multi-Layer Perceptron，MLP）的网络结构，有如下参数：

-   `x`：输入的张量
-   `sizes`：一个包含每一层的神经元个数的列表
-   `activation`：激活函数，用于中间层的神经元
-   `output_activation`：输出层的激活函数

该函数通过循环生成相应个数的全连接层，并将 `x` 作为输入传入。其中，`units` 指定每一层的神经元个数，`activation` 指定该层使用的激活函数，返回最后一层的结果。


    def mlp(x, sizes, activation=tf.tanh, output_activation=None):
        for size in sizes[:-1]:
            x = layers.Dense(units=size, activation=activation)(x)
        return layers.Dense(units=sizes[-1], activation=output_activation)(x)


（4）这里定义了一个函数 `logprobabilities`，用于计算给定动作 `a` 的对数概率。函数接受两个参数，`logits` 和 `a`，其中 `logits` 表示模型输出的未归一化的概率分布，`a` 表示当前采取的动作。函数首先对 `logits` 进行 softmax 归一化，然后对归一化后的概率分布取对数，得到所有动作的对数概率。接着，函数使用 `tf.one_hot` 函数生成一个 one-hot 编码的动作向量，并与所有动作的对数概率向量相乘，最后对结果进行求和得到给定动作的对数概率。

    def logprobabilities(logits, a):
        logprobabilities_all = tf.nn.log_softmax(logits)
        logprobability = tf.reduce_sum( tf.one_hot(a, num_actions) * logprobabilities_all, axis=1 )
        return logprobability


（5）这里定义了一个函数 `sample_action`。该函数接受一个 `observation`（观测值）参数，并在 actor 网络上运行该观测值以获得动作 logits（逻辑值）。然后使用逻辑值（logits）来随机采样出一个动作，并将结果作为函数的输出。


    @tf.function
    def sample_action(observation):
        logits = actor(observation)
        action = tf.squeeze(tf.random.categorical(logits, 1), axis=1)
        return logits, action

（6）这里定义了一个用于训练策略的函数`train_policy`。该函数使用带权重裁剪的 PPO 算法，用于更新 actor 的权重。

-   `observation_buffer`：输入的观测缓冲区
-   `action_buffer`：输入的动作缓冲区
-   `logprobability_buffer`：输入的对数概率缓冲区
-   `advantage_buffer`：输入的优势值缓冲区

在该函数内部，使用`tf.GradientTape`记录执行的操作，用于计算梯度并更新策略网络。计算的策略损失是策略梯度和剪裁比率的交集和。使用优化器`policy_optimizer`来更新actor的权重。最后，计算并返回 kl 散度的平均值，该值用于监控训练的过程。

 


    @tf.function
    def train_policy( observation_buffer, action_buffer, logprobability_buffer, advantage_buffer):
        with tf.GradientTape() as tape:   
            ratio = tf.exp( logprobabilities(actor(observation_buffer), action_buffer) - logprobability_buffer )
            min_advantage = tf.where(  advantage_buffer > 0, (1 + clip_ratio) * advantage_buffer, (1 - clip_ratio) * advantage_buffer, )
            policy_loss = -tf.reduce_mean( tf.minimum(ratio * advantage_buffer, min_advantage) )
        policy_grads = tape.gradient(policy_loss, actor.trainable_variables)
        policy_optimizer.apply_gradients(zip(policy_grads, actor.trainable_variables))
        kl = tf.reduce_mean( logprobability_buffer - logprobabilities(actor(observation_buffer), action_buffer) )
        kl = tf.reduce_sum(kl)
        return kl


（7）这里实现了价值函数（critic）的训练过程，函数接受两个参数：一个是 `observation_buffer`，表示当前存储的状态观察序列；另一个是 `return_buffer`，表示状态序列对应的回报序列。在函数内部，首先使用 `critic` 模型来预测当前状态序列对应的状态值（V）， 然后计算当前状态序列的平均回报与 V 之间的均方误差，并对其进行求和取平均得到损失函数 `value_loss`。接下来计算梯度来更新可训练的变量值。


    @tf.function
    def train_value_function(observation_buffer, return_buffer):
        with tf.GradientTape() as tape:  
            value_loss = tf.reduce_mean((return_buffer - critic(observation_buffer)) ** 2)
        value_grads = tape.gradient(value_loss, critic.trainable_variables)
        value_optimizer.apply_gradients(zip(value_grads, critic.trainable_variables))
        
## 游戏初始化

这里用于构建强化学习中的 Actor-Critic 网络模型。首先，使用 gy m库中的 CartPole-v0 环境创建一个环境实例 env 。然后，定义了两个变量，分别表示观测空间的维度 observation_dimensions 和动作空间的大小 num_actions，这些信息都可以从 env 中获取。接着，定义了一个 Buffer 类的实例，用于存储每个时间步的观测、动作、奖励、下一个观测和 done 信号，以便后面的训练使用。

然后，使用 Keras 库定义了一个神经网络模型 Actor ，用于近似模仿策略函数，该模型输入是当前的观测，输出是每个动作的概率分布的对数。

另外，还定义了一个神经网络模型 Critic ，用于近似模仿值函数，该模型输入是当前的观测，输出是一个值，表示这个观测的价值。最后，定义了两个优化器，policy_optimizer 用于更新 Actor 网络的参数，value_optimizer 用于更新 Critic 网络的参数。


    env = gym.make("CartPole-v0")
    observation_dimensions = env.observation_space.shape[0]
    num_actions = env.action_space.n
    buffer = Buffer(observation_dimensions, steps_per_epoch)

    observation_input = keras.Input(shape=(observation_dimensions,), dtype=tf.float32)
    logits = mlp(observation_input, list(hidden_sizes) + [num_actions], tf.tanh, None)
    actor = keras.Model(inputs=observation_input, outputs=logits)
    value = tf.squeeze( mlp(observation_input, list(hidden_sizes) + [1], tf.tanh, None), axis=1 )
    critic = keras.Model(inputs=observation_input, outputs=value)

    policy_optimizer = keras.optimizers.Adam(learning_rate=policy_learning_rate)
    value_optimizer = keras.optimizers.Adam(learning_rate=value_function_learning_rate)
    
    
## 保存未训练时的运动情况
   
在未训练模型之前，将模型控制游戏的情况保存是 gif ，可以看出来技术很糟糕，很快就结束了游戏。
  
    import imageio
    start = env.reset() 
    frames = []
    for t in range(steps_per_epoch):
        frames.append(env.render(mode='rgb_array'))
        start = start.reshape(1, -1)
        logits, action = sample_action(start)
        start, reward, done, _ = env.step(action[0].numpy())
        if done:
            break

    with imageio.get_writer('未训练前的样子.gif', mode='I') as writer:
        for frame in frames:
            writer.append_data(frame)
            
![未训练前的样子.gif](https://p9-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/63b79676ddf141f181711350876041d9~tplv-k3u1fbpfcp-watermark.image?)
            
## 模型训练

这里主要是训练模型，执行 eopch 轮，每一轮中循环  steps_per_epoch 步，每一步就是根据当前的观测结果 observation 来抽样得到下一步动作，然后将得到的各种观测结果、动作、奖励、value 值、对数概率值保存在 buffer 对象中，待这一轮执行游戏运行完毕，收集了一轮的数据之后，就开始训练策略和值函数，并打印本轮的训练结果，不断重复这个过程，

    observation, episode_return, episode_length = env.reset(), 0, 0
    for epoch in tqdm(range(epochs)):
        sum_return = 0
        sum_length = 0
        num_episodes = 0

        for t in range(steps_per_epoch):
            if render:
                env.render()

            observation = observation.reshape(1, -1)
            logits, action = sample_action(observation)
            observation_new, reward, done, _ = env.step(action[0].numpy())
            episode_return += reward
            episode_length += 1

            value_t = critic(observation)
            logprobability_t = logprobabilities(logits, action)

            buffer.store(observation, action, reward, value_t, logprobability_t)

            observation = observation_new

            terminal = done
            if terminal or (t == steps_per_epoch - 1):
                last_value = 0 if done else critic(observation.reshape(1, -1))
                buffer.finish_trajectory(last_value)
                sum_return += episode_return
                sum_length += episode_length
                num_episodes += 1
                observation, episode_return, episode_length = env.reset(), 0, 0

        ( observation_buffer, action_buffer, advantage_buffer,  return_buffer, logprobability_buffer, ) = buffer.get()

        for _ in range(train_policy_iterations):
            kl = train_policy( observation_buffer, action_buffer, logprobability_buffer, advantage_buffer )
            if kl > 1.5 * target_kl:
                break

        for _ in range(train_value_iterations):
            train_value_function(observation_buffer, return_buffer)

        print( f"完成第 {epoch + 1} 轮训练， 平均奖励: {sum_length / num_episodes}" )
        
打印：


    完成第 1 轮训练， 平均奖励: 30.864197530864196
    完成第 2 轮训练， 平均奖励: 40.32258064516129
    ...
    完成第 9 轮训练， 平均奖励: 185.1851851851852
    完成第 11 轮训练， 平均奖励: 172.41379310344828
    ...
    完成第 14 轮训练， 平均奖励: 172.41379310344828
    ...
    完成第 18 轮训练， 平均奖励: 185.1851851851852
    ...
    完成第 20 轮训练， 平均奖励: 200.0


## 保存训练后的运动情况

在训练模型之后，将模型控制游戏的情况保存是 gif ，可以看出来技术很娴熟，可以在很长的时间内使得棒子始终保持近似垂直的状态。

    import imageio
    start = env.reset()
    frames = []
    for t in range(steps_per_epoch):
        frames.append(env.render(mode='rgb_array'))
        start = start.reshape(1, -1)
        logits, action = sample_action(start)
        start, reward, done, _ = env.step(action[0].numpy())
        if done:
            break


    with imageio.get_writer('训练后的样子.gif', mode='I') as writer:
        for frame in frames:
            writer.append_data(frame)
            
            
![训练后的样子.gif](https://p3-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/35659d5ef94d4dbfa539d6039db2c168~tplv-k3u1fbpfcp-watermark.image?)