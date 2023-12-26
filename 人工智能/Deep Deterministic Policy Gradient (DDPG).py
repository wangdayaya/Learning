import gym
import tensorflow as tf
from keras import layers
import numpy as np
import matplotlib.pyplot as plt


for gpu in tf.config.experimental.list_physical_devices(device_type='gpu'):
    tf.config.experimental.set_memory_growth(gpu, True)

env = gym.make("Pendulum-v0")
num_states = env.observation_space.shape[0]  # 3
num_actions = env.action_space.shape[0]  # 1
upper_bound = env.action_space.high[0]
lower_bound = env.action_space.low[0]
print(f"状态空间大小 {num_states} , 动作空间大小 {num_actions} ，动作值上限 {upper_bound} ，动作之下限 {lower_bound} ")


class OUActionNoise:
    def __init__(self, mean, std_devivation, theta=0.15, dt=1e-2, x_initial=None):
        self.theta = theta
        self.mean = mean
        self.std_dev = std_devivation
        self.dt = dt
        self.x_initial = x_initial
        self.reset()

    def __call__(self):
        x = (self.x_prev
             + self.theta * (self.mean - self.x_prev) * self.dt
             + self.std_dev * np.sqrt(self.dt) * np.random.normal(size=self.mean.shape))
        self.x_prev = x
        return x

    def reset(self):
        if self.x_initial is not None:
            self.x_prev = self.x_initial
        else:
            self.x_prev = np.zeros_like(self.mean)


def get_actor():
    last_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)
    inputs = layers.Input(shape=(num_states,))
    out = layers.Dense(256, activation="relu")(inputs)
    out = layers.Dense(256, activation="relu")(out)
    outputs = layers.Dense(1, activation="tanh", kernel_initializer=last_init)(out)
    outputs = outputs * upper_bound
    model = tf.keras.Model(inputs, outputs)
    return model


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


ou_noise = OUActionNoise(mean=np.zeros(1), std_devivation=0.2 * np.ones(1))
actor_model = get_actor()
target_actor = get_actor()
critic_model = get_critic()
target_critic = get_critic()
target_actor.set_weights(actor_model.get_weights())
target_critic.set_weights(critic_model.get_weights())
critic_optimizer = tf.keras.optimizers.Adam(0.002)
actor_optimizer = tf.keras.optimizers.Adam(0.001)


class Buffer:
    def __init__(self, buffer_capacity=100000, batch_size=64):
        self.buffer_capacity = buffer_capacity
        self.batch_size = batch_size
        self.buffer_count = 0
        self.state_buffer = np.zeros((self.buffer_capacity, num_states))
        self.action_buffer = np.zeros((self.buffer_capacity, num_actions))
        self.reward_buffer = np.zeros((self.buffer_capacity, 1))
        self.next_state_buffer = np.zeros((self.buffer_capacity, num_states))

    def record(self, obs_tuple):
        index = self.buffer_count % self.buffer_capacity
        self.state_buffer[index] = obs_tuple[0]
        self.action_buffer[index] = obs_tuple[1]
        self.reward_buffer[index] = obs_tuple[2]
        self.next_state_buffer[index] = obs_tuple[3]
        self.buffer_count += 1

    @tf.function
    def update(self, state_batch, action_batch, reward_batch, next_state_batch):
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

    def learn(self):
        record_range = min(self.buffer_count, self.buffer_capacity)
        batch_indices = np.random.choice(record_range, self.batch_size)
        state_batch = tf.convert_to_tensor(self.state_buffer[batch_indices])
        action_batch = tf.convert_to_tensor(self.action_buffer[batch_indices])
        reward_batch = tf.convert_to_tensor(self.reward_buffer[batch_indices])
        reward_batch = tf.cast(reward_batch, dtype=tf.float32)
        next_state_batch = tf.convert_to_tensor(self.next_state_buffer[batch_indices])
        self.update(state_batch, action_batch, reward_batch, next_state_batch)


@tf.function
def update_target(target_weights, weights, tau):
    for (a, b) in zip(target_weights, weights):
        a.assign(b * tau + a * (1 - tau))


total_episodes = 100
gamma = 0.99
tau = 0.005
buffer = Buffer(50000, 64)
ep_reward_list = []
avg_reward_list = []


def policy(state, noise_object):
    sampled_actions = tf.squeeze(actor_model(state))
    noise = noise_object()
    sampled_actions = sampled_actions.numpy() + noise
    legal_action = np.clip(sampled_actions, lower_bound, upper_bound)
    return [np.squeeze(legal_action)]


for ep in range(total_episodes):
    prev_state = env.reset()
    episodic_reward = 0
    while True:
        tf_prev_state = tf.expand_dims(tf.convert_to_tensor(prev_state), 0)
        action = policy(tf_prev_state, ou_noise)
        state, reward, done, info = env.step(action)
        buffer.record((prev_state, action, reward, state))
        episodic_reward += reward
        buffer.learn()
        update_target(target_actor.variables, actor_model.variables, tau)
        update_target(target_critic.variables, critic_model.variables, tau)
        if done:
            break
        prev_state = state
    ep_reward_list.append(episodic_reward)
    avg_reward = np.mean(ep_reward_list[-40:])
    print("Episode * {} * Avg Reward is ==> {}".format(ep, avg_reward))
    avg_reward_list.append(avg_reward)

plt.plot(avg_reward_list)
plt.xlabel("Episode")
plt.ylabel("Avg Epsiodic reward")
plt.show()

# Episode * 0 * Avg
# Reward is == > -1157.8143517891501
# Episode * 1 * Avg
# Reward is == > -1189.9761222732382
# Episode * 2 * Avg
# Reward is == > -1044.89974715879
# ...
# Episode * 36 * Avg
# Reward is == > -557.5050289524168
# Episode * 37 * Avg
# Reward is == > -546.1769189528364
# Episode * 38 * Avg
# Reward is == > -535.4149222739205
# ...
# Episode * 97 * Avg
# Reward is == > -183.08817536189082
# Episode * 98 * Avg
# Reward is == > -183.19096335666998
# Episode * 99 * Avg
# Reward is == > -183.10935571438955