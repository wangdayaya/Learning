

##  CartPole 介绍

在一个光滑的轨道上有个推车，杆子垂直微置在推车上，随时有倒的风险。系统每次对推车施加向左或者向右的力，但我们的目标是让杆子保持直立。杆子保持直立的每个时间单位都会获得 +1 的奖励。但是当杆子与垂直方向成 15 度以上的位置，或者推车偏离中心点超过 2.4 个单位后，这一轮局游戏结束。因此我们可以获得的最高回报等于 200 。我们这里就是要通过使用 PPO 算法来训练一个强化学习模型 actor-critic ，通过对比模型训练前后的游戏运行 gif 图，可以看出来我们训练好的模型能长时间保持杆子处于垂直状态。

## Actor Critic 介绍

当 agent 采取行动并在环境中移动时，它在观察到的环境状态的情况下，学习两个可能的输出：

- 接下来最合适的一个操作，actor 负责此部分输出。
- 未来可能获得的奖励总和，critic 负责此部分的输出。

actor 和 critic 通过不断地学习，以便使得 agent 在游戏中最终获得的奖励最大，这里的 agent 就是那个小车。
 
## 库准备

    tensorflow-gpu==2.10.0
    imageio==2.26.1
    keras==2.10,0
    gym==0.20.0
    pyglet==1.5.20
    scipy==1.10.1

## 设置超参数

这部分代码主要有：

（1）导入所需的Python库：gym、numpy、tensorflow 和 keras。

（2）设置整个环境的超参数：种子、折扣因子和每个回合的最大步数。

（3）创建 CartPole-v0 环境，并设置种子。

（4）定义一个非常小的值 eps ，表示的机器两个不同的数字之间的最小差值,用于检验数值稳定性。

    import gym # 导入Gym库，用于开发和比较强化学习算法
    import numpy as np # 导入NumPy库，用于进行科学计算
    import tensorflow as tf # 导入TensorFlow库
    from tensorflow import keras # 导入keras模块，这是一个高级神经网络API
    from tensorflow.keras import layers # 导入keras中的layers模块，用于创建神经网络层

    seed = 42 # 设定随机种子，用于复现实验结果
    gamma = 0.99 # 定义折扣率，用于计算未来奖励的现值
    max_steps_per_episode = 10000 # 设定每个 episode 的最大步数
    env = gym.make("CartPole-v0") # 创建 CartPole-v0 环境实例
    env.seed(seed) # 设定环境的随机种子
    eps = np.finfo(np.float32).eps.item() # 获取 float32 数据类型的误差最小值 epsilon 

## Actor Critic 结构搭建


（1）Actor：将环境的状态作为输入，返回操作空间中每个操作及其概率值，其实总共只有两个操作，往左和往右。

（2）Critic：将环境的状态作为输入，返回未来奖励综合的估计。

（3）在这里网络结构中我们在一开始接收 inputs 之后，我们的 Actor 和 Critic 共用了中间的部分隐藏层 common 层，然后在一个输出分支上连接了一个全连接进行动作分类作为 action ，另一个分支上连接了一个全连接层进行未来奖励计算作为 critic 。

    num_inputs = 4 # 状态空间的维度，即输入层的节点数
    num_actions = 2 # 行为空间的维度，即输出层的节点数
    num_hidden = 128 # 隐藏层的节点数

    inputs = layers.Input(shape=(num_inputs,)) # 创建输入层，指定输入的形状
    common = layers.Dense(num_hidden, activation="relu")(inputs) # 创建一个全连接层，包含num_hidden 个神经元，使用 ReLU 作为激活函数
    action = layers.Dense(num_actions, activation="softmax")(common) # 创建一个全连接层，包含 num_actions 个神经元，使用 softmax 作为激活函数
    critic = layers.Dense(1)(common) # 创建一个全连接层，包含1个神经元

    model = keras.Model(inputs=inputs, outputs=[action, critic]) # 创建一个 Keras 模型，包含输入层、共享的隐藏层和两个输出层



## 训练前的样子

    import imageio
    start = env.reset() 
    frames = []
    for t in range(max_steps_per_episode):
        frames.append(env.render(mode='rgb_array'))
        start = start.reshape(1, -1)
        start, reward, done, _ = env.step(np.random.choice(num_actions, p=np.squeeze(action_probs)))
        if done:
            break

    with imageio.get_writer('未训练前的样子.gif', mode='I') as writer:
        for frame in frames:
            writer.append_data(frame)


![未训练前的样子.gif](https://p9-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/5a2c68eca7c44da092ae890599d0a229~tplv-k3u1fbpfcp-watermark.image?)

## 模型训练

设置训练所需要的优化器，以及各种参数来记录每个时间步上的数据。

    optimizer = keras.optimizers.Adam(learning_rate=0.01) # 创建 Adam 优化器实例，设置学习率为 0.01
    huber_loss = keras.losses.Huber() # 创建损失函数实例
    action_probs_history = [] # 创建一个列表，用于保存 action 网络在每个步骤中采取各个行动的概率
    critic_value_history = [] # 创建一个列表，用于保存 critic 网络在每个步骤中对应的值
    rewards_history = [] # 创建一个列表，用于保存每个步骤的奖励值
    running_reward = 0 # 初始化运行过程中的每轮奖励
    episode_count = 0 # 初始化 episode 计数器

一直训练下去，直到满足奖励大于 195 才会停下训练过程。

    while True:  
        state = env.reset()  # 新一轮游戏开始，重置环境
        episode_reward = 0  # 记录本轮游戏的总奖励值
        with tf.GradientTape() as tape:  # 构建 GradientTape 用于计算梯度
            for timestep in range(1, max_steps_per_episode): # 本轮游戏如果一切正常会进行 max_steps_per_episode 步
                state = tf.convert_to_tensor(state)  # 将状态转换为张量
                state = tf.expand_dims(state, 0)  # 扩展维度，以适应模型的输入形状

                action_probs, critic_value = model(state)  # 前向传播，得到 action 网络输出的动作空间的概率分布，和 critic 网络预测的奖励值
                critic_value_history.append(critic_value[0, 0])  # 将上面 critic 预测的奖励值记录在 critic_value_history 列表中

                action = np.random.choice(num_actions, p=np.squeeze(action_probs))  # 依据概率分布抽样某个动作，当然了某个动作概率越大越容易被抽中，同时也保留了一定的随机性
                action_probs_history.append(tf.math.log(action_probs[0, action]))  # 将使用该动作的对数概率值记录在 action_probs_history 列表中

                state, reward, done, _ = env.step(action)  # 游戏环境使用选中的动作去执行，得到下一个游戏状态、奖励、是否终止和其他信息
                rewards_history.append(reward)  # 将该时刻的奖励记录在 rewards_history 列表中
                episode_reward += reward  # 累加本轮游戏的总奖励值

                if done:  # 如果到达终止状态，则结束循环
                    break

            running_reward = 0.05 * episode_reward + (1 - 0.05) * running_reward  # 计算平均奖励

            returns = []  # 存储折扣回报
            discounted_sum = 0
            for r in rewards_history[::-1]:  # 从后往前遍历奖励的历史值
                discounted_sum = r + gamma * discounted_sum  # 计算折扣回报
                returns.insert(0, discounted_sum)  # 将折扣回报插入列表的开头，最后形成的还是从前往后的折扣奖励列表

            returns = np.array(returns)  # 将折扣回报转换为数组
            returns = (returns - np.mean(returns)) / (np.std(returns) + eps)  # 归一化折扣回报
            returns = returns.tolist()  # 将折扣回报转换为列表形式

            history = zip(action_probs_history, critic_value_history, returns)  # 将三个列表进行 zip 压缩
            actor_losses = []  # 存储 action 网络的损失
            critic_losses = []  # 存储 critic 网络的损失

            for log_prob, value, ret in history:
                diff = ret - value
                actor_losses.append(-log_prob * diff)  # 计算 actor 的损失函数

                critic_losses.append(
                    huber_loss(tf.expand_dims(value, 0), tf.expand_dims(ret, 0)) # 计算 critic 的损失函数
                )

            loss_value = sum(actor_losses) + sum(critic_losses) # 计算总损失函数
            grads = tape.gradient(loss_value, model.trainable_variables) # 计算梯度
            optimizer.apply_gradients(zip(grads, model.trainable_variables)) # 更新模型参数

            action_probs_history.clear() # 清空之前的历史记录
            critic_value_history.clear() # 清空之前的历史记录
            rewards_history.clear() # 清空之前的历史记录

        episode_count += 1 # 当一轮游戏结束时， episode 加一
        if episode_count % 10 == 0: # 每训练 10 个 episode ，输出当前的平均奖励
            template = "在第 {} 轮游戏中获得奖励: {:.2f} 分"
            print(template.format(episode_count, running_reward))

        if running_reward > 195:  # 如果平均奖励超过195，视为任务已经解决
            print("奖励超过 195 ，训练结束")
            break

打印：

```
在第 10 轮游戏中获得奖励: 11.17 分
在第 20 轮游戏中获得奖励: 17.12 分
...
在第 170 轮游戏中获得奖励: 155.02 分
在第 180 轮游戏中获得奖励: 171.67 分
...
在第 220 轮游戏中获得奖励: 193.74 分
奖励超过 195 ，训练结束
```

## 训练后的样子


    import imageio
    start = env.reset() 
    frames = []
    for t in range(max_steps_per_episode):
        frames.append(env.render(mode='rgb_array'))
        start = start.reshape(1, -1)
        action_probs, _ = model(start)
        action = np.random.choice(num_actions, p=np.squeeze(action_probs))
        start, reward, done, _ = env.step(action)
        if done:
            break

    with imageio.get_writer('训练后的样子.gif', mode='I') as writer:
        for frame in frames:
            writer.append_data(frame)

![训练后的样子.gif](https://p6-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/7dc2a501e96241c0bce11a46cb023727~tplv-k3u1fbpfcp-watermark.image?)