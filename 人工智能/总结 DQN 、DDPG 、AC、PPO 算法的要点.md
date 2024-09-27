# Deep Q-Learning ，DQN（off-policy）

代码详见：https://keras.io/examples/rl/deep_q_network_breakout/

1. 创建两个一样的模型，一个是 model ，一个是 model_target ，输入都是状态画面，中间是图像卷积处理，输出都是动作空间大小维度的实数作为奖励期望

        def create_q_model():
            # Network defined by the Deepmind paper
            return keras.Sequential(
                [
                    layers.Lambda(
                        lambda tensor: keras.ops.transpose(tensor, [0, 2, 3, 1]),
                        output_shape=(84, 84, 4),
                        input_shape=(4, 84, 84),
                    ),
                    # Convolutions on the frames on the screen
                    layers.Conv2D(32, 8, strides=4, activation="relu", input_shape=(4, 84, 84)),
                    layers.Conv2D(64, 4, strides=2, activation="relu"),
                    layers.Conv2D(64, 3, strides=1, activation="relu"),
                    layers.Flatten(),
                    layers.Dense(512, activation="relu"),
                    layers.Dense(num_actions, activation="linear"),
                ]
            )
        model = create_q_model()
        # 为了稳定训练
        model_target = create_q_model()


    目标网络是当前策略网络和价值网络的延迟复制品，它们的权重会在一定步数后更新，使用目标网络的原因如下：

- 稳定性：使用目标网络可以提高训练过程的稳定性。在Actor-Critic算法中，评论家（Critic）用于估计状态或状态-动作对的价值。如果评论家的估计受到当前策略的较大影响，那么它可能会对策略的波动过于敏感，导致训练不稳定。

- 减少方差：目标网络提供了一个更平滑的目标值，这有助于减少估计的方差。在计算损失时，目标网络的输出不会随着当前网络的输出变化，这有助于稳定评论家的损失函数。

- 避免过度乐观：在没有目标网络的情况下，评论家可能会过于乐观地估计动作的价值，因为它总是使用当前策略的输出。目标网络提供了一个更保守的估计，有助于避免这种过度乐观的偏差。

- 异步更新：目标网络的权重是定期更新的，而不是实时更新。这种异步更新策略有助于稳定训练过程，因为它避免了目标价值的频繁变化。


2.  epsilon-greedy for exploration
在执行所有 step 的时候， epsilon 需要不断均匀减小，直到最小保持在 epsilon_min=0.1 ，在每一步 step 当概率小于 epsilon 的时候采用随机的探索动作，否则直接使用模型采用分数最大的动作

        if epsilon > np.random.rand(1)[0]:
            action = np.random.choice(num_actions)
        else:
            state_tensor = keras.ops.convert_to_tensor(state)
            state_tensor = keras.ops.expand_dims(state_tensor, 0)
            action_probs = model(state_tensor, training=False)
            action = keras.ops.argmax(action_probs[0]).numpy()
        epsilon -= epsilon_interval / epsilon_greedy_frames
        epsilon = max(epsilon, epsilon_min)


3. 每个 epoch 从开始执行所有的 step 直到结束，将每一步的 step 数据存入历史数组，每隔 update_after_actions 个 step 使用经验回放技术，从历史数据中采样得到 batch_size 个样本进行模型训练


4.  训练细节


        # 使用 model_target ，为了训练稳定
        future_rewards = model_target.predict(state_next_sample)
        # 取 future_rewards 中每个 batch 的最大值当作预期奖励
        updated_q_values = rewards_sample + gamma * keras.ops.amax( future_rewards, axis=1 )
        masks = keras.ops.one_hot(action_sample, num_actions)
        with tf.GradientTape() as tape:
            # 使用 model
            q_values = model(state_sample)
            q_action = keras.ops.sum(keras.ops.multiply(q_values, masks), axis=1)
            # 使用 keras.losses.Huber() 或者 MSE 作为损失
            loss = loss_function(updated_q_values, q_action)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))



5. 每隔 update_target_network  个 step 就更新一次  model_target ，直接将 model 的权重赋值即可

6. 某个 epoch 中所有 step 获得的系统 reward 都累加起来作为当前 epoch 的奖励存入列表 episode_reward_history ，只选择 episode_reward_history 最后 100 个的 epoch 的奖励总和平均值作为我们的训练过程模型可以得到的得分，以此来当作模型是否训练好的指标，如果达到了预定的指标就提前结束训练。




# Deep Deterministic Policy Gradient，DDPG （off-policy）

代码详见：https://keras.io/examples/rl/ddpg_pendulum/

1. actor-critic 结构，但是因为要处理的是连续动作而非离散动作，所以结构和标准的 actor-critic 结构略有不同，actor 根据状态给出执行的 -2 到 +2 之间的动作实数，critic 根据状态特征和 actor 给出的动作拼接起来的特征，输出一个打分结果，本质上像是一个动作价值函数 Q(s,a)。另外各有一个相同的副本模型，actor_target 和 critic_target ，初始化的时候直接复制即可。

        def get_actor():
            last_init = keras.initializers.RandomUniform(minval=-0.003, maxval=0.003)
            inputs = layers.Input(shape=(num_states,))
            out = layers.Dense(256, activation="relu")(inputs)
            out = layers.Dense(256, activation="relu")(out)
            outputs = layers.Dense(1, activation="tanh", kernel_initializer=last_init)(out)
            outputs = outputs * upper_bound
            model = keras.Model(inputs, outputs)
            return model

        def get_critic():
            state_input = layers.Input(shape=(num_states,))
            state_out = layers.Dense(16, activation="relu")(state_input)
            state_out = layers.Dense(32, activation="relu")(state_out)
            action_input = layers.Input(shape=(num_actions,))
            action_out = layers.Dense(32, activation="relu")(action_input)
            concat = layers.Concatenate()([state_out, action_out])
            out = layers.Dense(256, activation="relu")(concat)
            out = layers.Dense(256, activation="relu")(out)
            outputs = layers.Dense(1)(out)
            model = keras.Model([state_input, action_input], outputs)
            return model

        actor_model = get_actor()
        critic_model = get_critic()
        target_actor = get_actor()
        target_critic = get_critic()
        target_actor.set_weights(actor_model.get_weights())
        target_critic.set_weights(critic_model.get_weights())

2. 在每个 step 中通过 actor 获取输出的预测动作的时候要加入一定的噪声为了能更好的进行未知动作的探索，这里的噪声使用的 OUActionNoise 算法。

        def policy(state, noise_object):
            sampled_actions = keras.ops.squeeze(actor_model(state))
            noise = noise_object()
            # Adding noise to action
            sampled_actions = sampled_actions.numpy() + noise
            # We make sure action is within bounds
            legal_action = np.clip(sampled_actions, lower_bound, upper_bound)
            return [np.squeeze(legal_action)]

3. 一共执行 total_episodes 个 epoch ，每个 epoch 从头开始执行每个 step ，每个 step 都往经验数组中存放一条数据，并且每个 step 都要从经验数据中随机抽取 batch_size 条数据进行训练 actor_model 和 critic_model  ，然后还需要更新 target_actor 和 target_critic  ，需要注意的是在计算 y 的时候使用的是 target_actor 和 target_critic ，因为每步都要更新 critic_model 和 actor_model 可能会导致训练不稳定，使用 target_actor 和 target_critic 有助于稳定训练。

        def update( self, state_batch, action_batch, reward_batch, next_state_batch, ):
            with tf.GradientTape() as tape:
                target_actions = target_actor(next_state_batch, training=True)
                y = reward_batch + gamma * target_critic( [next_state_batch, target_actions], training=True )
                critic_value = critic_model([state_batch, action_batch], training=True)
                critic_loss = keras.ops.mean(keras.ops.square(y - critic_value))

            critic_grad = tape.gradient(critic_loss, critic_model.trainable_variables)
            critic_optimizer.apply_gradients( zip(critic_grad, critic_model.trainable_variables) )

            with tf.GradientTape() as tape:
                # 这里不需要使用 target 网络，因为我们已经拿到了目前最好的 critic ，根据它的“准确打分”来优化 actor_model 即可
                actions = actor_model(state_batch, training=True)
                critic_value = critic_model([state_batch, actions], training=True)
                # Used `-value` as we want to maximize the value given by the critic for our actions
                actor_loss = -keras.ops.mean(critic_value)

            actor_grad = tape.gradient(actor_loss, actor_model.trainable_variables)
            actor_optimizer.apply_gradients( zip(actor_grad, actor_model.trainable_variables) )

4. 每个 step 训练模型结束之后，同时要更新一下两个 target 模型，将新旧模型分别用0.05和0.95的比例进行加权求和即可，保证 target 模型的稳定性。

        def update_target(target, original, tau=0.05):
            target_weights = target.get_weights()
            original_weights = original.get_weights()
            for i in range(len(target_weights)):
                target_weights[i] = original_weights[i] * tau + target_weights[i] * (1 - tau)
            target.set_weights(target_weights)

5. 某个 epoch 中将每一步得到的 reward 都累加起来当作本轮的奖励存入 ep_reward_list ，取最后 40 个 epoch 的奖励的平均值作为我们的训练过程模型可以得到的得分，以此来当作模型是否训练好的指标，如果达到了预定的指标就提前结束训练。



# Actor-Critic Method ，基于值函数方法和基于策略函数方法的叠加

代码详见：https://keras.io/examples/rl/actor_critic_cartpole/

actor：动作空间中每个动作的概率值。 
critic：预计在未来收到的所有奖励的总和。 


1. actor-critic 结构 ，共同享有输入的状态特征，actor 根据当前的状态返回动作概率分布，critic 根据当前的状态返回预期的奖励，也就是状态价值

        inputs = layers.Input(shape=(num_inputs,))
        common = layers.Dense(num_hidden, activation="relu")(inputs)
        action = layers.Dense(num_actions, activation="softmax")(common)
        critic = layers.Dense(1)(common)
        model = keras.Model(inputs=inputs, outputs=[action, critic])


2. 一次性执行完 epoch 中的所有 step ，将每个 step 的结果存入数组中在当前 epoch 结束之后，使用本次 epoch 的所有的数据更新 a2c 模型。将所有 step 的 reward 都累加起来当作本轮 epoch 的奖励值 episode_reward ，并且在每个 epoch 结束之后更新 running_reward ，running_reward = 0.05 * episode_reward + (1 - 0.05) * running_reward ，不断经历多个 epoch ，直到 running_reward 达到预定指标后，结束训练。

        for timestep in range(1, max_steps_per_episode):
            state = ops.convert_to_tensor(state)
            state = ops.expand_dims(state, 0)
            action_probs , critic_value = model(state)
            critic_value_history.append(critic_value[0, 0])
            action = np.random.choice(num_actions, p=np.squeeze(action_probs))
            action_probs_history.append(ops.log(action_probs[0, action]))
            state, reward, done, _ = env.step(action)
            rewards_history.append(reward)
            episode_reward += reward
            if done:
                break
        running_reward = 0.05 * episode_reward + (1 - 0.05) * running_reward
        returns = []
        discounted_sum = 0
        for r in rewards_history[::-1]:
            discounted_sum = r + gamma * discounted_sum
            returns.insert(0, discounted_sum)
        returns = np.array(returns)
        returns = (returns - np.mean(returns)) / (np.std(returns) + eps)
        returns = returns.tolist()
        history = zip(action_probs_history, critic_value_history, returns)
        actor_losses = []
        critic_losses = []
        for log_prob, value, ret in history:
            # ret，折扣回报，其实就是每个 step 的真实的未来加权奖励得分
            # value，模型预测回报，或者说是每个 step 预测的未来加权奖励得分
            diff = ret - value
            actor_losses.append(-log_prob * diff)  # actor loss
            critic_losses.append( huber_loss(ops.expand_dims(value, 0), ops.expand_dims(ret, 0)) )
        loss_value = sum(actor_losses) + sum(critic_losses)
        grads = tape.gradient(loss_value, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))



# Proximal Policy Optimization，PPO（on-policy）

代码详见：https://keras.io/examples/rl/ppo_cartpole/

1. 仍然使用 actor-critic  结构 ，actor 会根据状态返回动作空间大小的 logits ，critic 会根据状态返回 1 个实数，也就是状态价值

        observation_input = keras.Input(shape=(observation_dimensions,), dtype="float32")
        logits = mlp(observation_input, list(hidden_sizes) + [num_actions])
        actor = keras.Model(inputs=observation_input, outputs=logits)
        value = keras.ops.squeeze(mlp(observation_input, list(hidden_sizes) + [1]), axis=1)
        critic = keras.Model(inputs=observation_input, outputs=value)

2. 训练 actor 和 critic ，一次性执行完当前 epoch 的所有 step ，将所有数据都存储起来，然后一次性都取出来反复进行 train_iterations 次来全部用来更新模型 train_policy 和  train_value_function ，核心就是计算出优势值和回报值。


        def finish_trajectory(self, last_value=0):
            # Finish the trajectory by computing advantage estimates and rewards-to-go
            path_slice = slice(self.trajectory_start_index, self.pointer)
            rewards = np.append(self.reward_buffer[path_slice], last_value)
            values = np.append(self.value_buffer[path_slice], last_value)
            # 这里是计算优势，也就是 Q - V ，表示在某个状态下采取该行动与该状态的平均值相比有多好，优势计算的是每个 step 的优势，和使用全部 reward 的折扣回报不同，所以和上面 diff = ret - value 原理不同
            deltas = rewards[:-1] + self.gamma * values[1:] - values[:-1]
            # 需要进行一定的权重 self.gamma * self.lam 衰减，表示当前的优势对未来的预期回报影响越来越小
            self.advantage_buffer[path_slice] = discounted_cumulative_sums(deltas, self.gamma * self.lam)
            # 需要进行一定的权重 self.gamma 衰减，表示当前的 reward 对未来的预期回报影响越来越小
            self.return_buffer[path_slice] = discounted_cumulative_sums( rewards, self.gamma )[:-1]
            self.trajectory_start_index = self.pointer

        def train_policy( observation_buffer, action_buffer, logprobability_buffer, advantage_buffer ):
            with tf.GradientTape() as tape:  
                # 其实就是新旧两个策略函数的比值 
                ratio = keras.ops.exp( logprobabilities(actor(observation_buffer), action_buffer) - logprobability_buffer )
                min_advantage = keras.ops.where( advantage_buffer > 0,  (1 + clip_ratio) * advantage_buffer, (1 - clip_ratio) * advantage_buffer, )
                # 策略损失前的负号（-）是因为在优化过程中，我们希望最小化损失函数。由于我们的目标是最大化策略的预期回报，所以我们通过最小化负的预期回报来实现这一点。换句话说，最小化负的回报等价于最大化回报。
                policy_loss = -keras.ops.mean( keras.ops.minimum(ratio * advantage_buffer, min_advantage) )
            policy_grads = tape.gradient(policy_loss, actor.trainable_variables)
            policy_optimizer.apply_gradients(zip(policy_grads, actor.trainable_variables))
            kl = keras.ops.mean( logprobability_buffer - logprobabilities(actor(observation_buffer), action_buffer) )
            kl = keras.ops.sum(kl)
            return kl

        def train_value_function(observation_buffer, return_buffer):
            with tf.GradientTape() as tape:  # Record operations for automatic differentiation.
                value_loss = keras.ops.mean((return_buffer - critic(observation_buffer)) ** 2)
            value_grads = tape.gradient(value_loss, critic.trainable_variables)
            value_optimizer.apply_gradients(zip(value_grads, critic.trainable_variables))