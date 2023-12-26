import numpy as np
from baselines.common.atari_wrappers import make_atari, wrap_deepmind
import tensorflow as tf
from tensorflow import keras
from keras import layers

# git clone https://github.com/openai/baselines.git
# git clone https://github.com/openai/atari-py
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

seed = 42
gamma = 0.99
epsilon = 1.0
epsilon_min = 0.1
epsilon_max = 1.0
epsilon_interval = epsilon_max - epsilon_min
batch_size = 8
max_steps_per_episode = 10000
env = make_atari("BreakoutNoFrameskip-v4")
env = wrap_deepmind(env, frame_stack=True, scale=True)
env.seed(seed)
num_actions = 4


def create_q_model():
    inputs = layers.Input(shape=(84, 84, 4))
    layer1 = layers.Conv2D(32, 8, strides=4, activation="relu")(inputs)
    layer2 = layers.Conv2D(64, 4, strides=2, activation="relu")(layer1)
    layer3 = layers.Conv2D(64, 3, strides=1, activation="relu")(layer2)
    layer4 = layers.Flatten()(layer3)
    layer5 = layers.Dense(512, activation="relu")(layer4)
    action = layers.Dense(num_actions, activation="linear")(layer5)
    return keras.Model(inputs=inputs, outputs=action)


model = create_q_model()
model_target = create_q_model()

optimizer = keras.optimizers.Adam(learning_rate=0.00025, clipnorm=1.0)
action_history = []
state_history = []
state_next_history = []
rewards_history = []
done_history = []
episode_reward_history = []
running_reward = 0
episode_count = 0
frame_count = 0
epsilon_random_frames = 50000
epsilon_greedy_frames = 1000000.0
max_memory_length = 100000
update_after_actions = 4
update_target_network = 10000
loss_function = keras.losses.Huber()

while True:
    state = np.array(env.reset())
    episode_reward = 0
    for timestep in range(1, max_steps_per_episode):
        frame_count += 1
        if frame_count < epsilon_random_frames or epsilon > np.random.rand(1)[0]:
            action = np.random.choice(num_actions)
        else:
            state_tensor = tf.convert_to_tensor(state)
            state_tensor = tf.expand_dims(state_tensor, 0)
            action_probs = model(state_tensor, training=False)
            action = tf.argmax(action_probs[0]).numpy()
        epsilon -= epsilon_interval / epsilon_greedy_frames
        epsilon = max(epsilon, epsilon_min)
        state_next, reward, done, _ = env.step(action)
        state_next = np.array(state_next)
        episode_reward += reward
        action_history.append(action)
        state_history.append(state)
        state_next_history.append(state_next)
        done_history.append(done)
        rewards_history.append(reward)

        state = state_next
        if frame_count % update_after_actions == 0 and len(done_history) > batch_size:
            indices = np.random.choice(range(len(done_history)), size=batch_size)
            state_sample = np.array([state_history[i] for i in indices])
            state_next_sample = np.array([state_next_history[i] for i in indices])
            rewards_sample = np.array([rewards_history[i] for i in indices])
            action_sample = [action_history[i] for i in indices]
            done_sample = tf.convert_to_tensor([float(done_history[i]) for i in indices])
            future_rewards = model_target.predict(state_next_sample)
            update_q_values = rewards_sample + gamma * tf.reduce_max(future_rewards, axis=1)
            update_q_values = update_q_values * (1 - done_sample) - done_sample
            masks = tf.one_hot(action_sample, num_actions)
            with tf.GradientTape() as tape:
                q_values = model(state_sample)
                q_action = tf.reduce_sum(tf.multiply(q_values, masks), axis=1)
                loss = loss_function(update_q_values, q_action)
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

        if frame_count % update_target_network == 0:
            model_target.set_weights(model.get_weights())
            print("running reward: {:.2f} at episode {}, frame count {}".format(running_reward, episode_count,
                                                                                frame_count))

        if len(rewards_history) > max_memory_length:
            del rewards_history[:1]
            del state_history[:1]
            del state_next_history[:1]
            del action_history[:1]
            del done_history[:1]
        if done:
            break
    episode_reward_history.append(episode_reward)
    if len(episode_reward_history) > 100:
        del episode_reward_history[:1]
    running_reward = np.mean(episode_reward_history)
    episode_count += 1
    if running_reward > 40:
        print("Solved at {}".format(episode_count))
        break
