
import tensorflow as tf
import numpy as np
from tensorflow import keras
from collections import deque
import random

from snake_game import SnakeGame
from tensorflow.keras.layers import Conv2D, Flatten, MaxPooling2D

import matplotlib.pyplot as plt
import datetime
# tensorboard --logdir logs/fit
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
writer = tf.summary.create_file_writer(log_dir)

def agent(state_shape, action_shape):
    learning_rate = 0.001
    #init = tf.keras.initializers.HeUniform()
    model = keras.Sequential()
    model.add(Conv2D(6, (3, 3), input_shape=state_shape, activation='relu'))
    model.add(MaxPooling2D(2, 2))
    model.add(Conv2D(12, (3, 3), activation='relu'))
    model.add(MaxPooling2D(2, 2))
    model.add(Conv2D(24, (3, 3), activation='relu'))
    model.add(MaxPooling2D(1, 1))
    model.add(Flatten())
    model.add(keras.layers.Dense(24, activation='relu'))
    model.add(keras.layers.Dense(12, activation='relu'))
    model.add(keras.layers.Dense(action_shape, activation='linear'))
    model.compile(loss=tf.keras.losses.MeanSquaredError(),
                  optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  metrics=['accuracy'])
    model.summary()
    return model


def train(env, replay_memory, model, target_model, done, n):
    discount_factor = 0.9
    batch_size = 256
    mini_batch = random.sample(replay_memory, batch_size)
    current_states = np.array([transition[0] for transition in mini_batch])
    current_qs_list = model.predict(current_states)
    new_current_states = np.array([transition[3] for transition in mini_batch])
    future_qs_list = target_model.predict(new_current_states)
    X = []
    Y = []
    for index, (observation, action, reward, new_observation, done) in enumerate(mini_batch):
        if not done:
            max_future_q = reward + discount_factor * np.max(future_qs_list[index])
        else:
            max_future_q = reward
        current_qs = current_qs_list[index]
        current_qs[action] = max_future_q
        X.append(observation)
        Y.append(current_qs)
    history = model.fit(np.array(X), np.array(Y), batch_size=batch_size, verbose=0, shuffle=True)
    with writer.as_default():
        tf.summary.scalar("loss", history.history['loss'][0], step=n)


def heuristic(env):
    score, apple, head, tail, direction = env.get_state()
    d_row = apple[0][0] - head[0]
    d_col = apple[0][1] - head[1]
    dir_aux = 0
    act = 0
    if abs(d_row) > abs(d_col):
        if d_row < 0:
            dir_aux = 0
        else:
            dir_aux = 2
    else:
        if d_col < 0:
            dir_aux = 3
        else:
            dir_aux = 1
    act = dir_aux - direction
    if act > 1:
        act = 1
    if act < -1:
        act = -1
    return act


def plot_board(file_name, board, text=None):
    plt.figure(figsize=(10, 10))
    plt.imshow(board)
    plt.axis('off')
    if text is not None:
        plt.gca().text(3, 3, text, fontsize=45, color='yellow')
    plt.savefig(file_name, bbox_inches='tight')
    plt.close()


def main(try_number, weights=None):
    actions = 3  # left, right, straight
    rgb = 3
    train_episodes = 4096
    epsilon = 1
    max_epsilon = 1
    min_epsilon = 0.1
    decay = 0.001
    MIN_REPLAY_SIZE = 1000
    total_steps = []
    total_rewards = []
    epsilons = []

    if try_number == 1:
        width = 14
        height = 14
        border = 10
        env = SnakeGame(14, 14, food_amount=1, border=10)
    elif try_number == 2:
        width = 14
        height = 14
        border = 10
        env = SnakeGame(14, 14, food_amount=4, border=10)
    elif try_number == 3:
        width = 30
        height = 30
        border = 1
        env = SnakeGame(30, 30, food_amount=4, border=1)
    else:
        width = 30
        height = 30
        border = 1
        env = SnakeGame(30, 30, border=1)

    model = agent((width + border * 2, height + border * 2, rgb), actions)
    target_model = agent((width + border * 2, height + border * 2, rgb), actions)
    if try_number > 1:
        model.set_weights(weights)

    target_model.set_weights(model.get_weights())

    replay_memory = deque(maxlen=100000)
    steps_to_update_target_model = 0
    observation, reward, done, info = env.reset()

    pos_rew = 0
    neutral_rew = 0
    neg_rew = 0

    while len(replay_memory) < 99999:
        if random.random() <= 0.7:
            action = heuristic(env)
            aux = action
        else:
            action = random.randint(0, 2)
            aux = action - 1
        new_observation, reward, done, info = env.step(aux)
        if reward >= 1:
            replay_memory.append([observation, action, reward, new_observation, done])
            pos_rew += 1
            observation = new_observation
        elif done:
            replay_memory.append([observation, action, reward, new_observation, done])
            observation, reward, done, info = env.reset()
            neg_rew += 1
        elif random.random() <= 0.1:
            replay_memory.append([observation, action, reward, new_observation, done])
            neutral_rew += 1
            observation = new_observation

    print(pos_rew)
    print(neutral_rew)
    print(neg_rew)

    for episode in range(train_episodes):
        print("episode " + str(episode))
        total_training_rewards = 0
        total_score = 0
        observation, reward, done, info = env.reset()
        done = False
        steps_to_train = 0
        while not done:
            steps_to_train += 1
            steps_to_update_target_model += 1
            random_number = np.random.rand()
            if random_number <= epsilon:
                action = random.randint(0, 2)
            else:
                reshaped = observation.reshape([1, observation.shape[0], observation.shape[1], observation.shape[2]])
                predicted = model.predict(reshaped).flatten()
                action = np.argmax(predicted)
            aux = action - 1
            new_observation, reward, done, info = env.step(aux)

            if reward >= 1:
                total_score += 1

            replay_memory.append([observation, action, reward, new_observation, done])
            if len(replay_memory) >= MIN_REPLAY_SIZE and \
                    (steps_to_train % 256 == 0 or done):
                train(env, replay_memory, model, target_model, done, episode)

            observation = new_observation
            total_training_rewards += reward
            if done:
                total_steps.append(steps_to_train)
                total_training_rewards += 1
                print('Rewards: {} after n steps = {}; score = {}'.format(
                    total_training_rewards, steps_to_train, total_score))
                total_rewards.append(total_score)

                if steps_to_update_target_model >= 100:
                    print('Copying main network weights to the target network weights')
                    target_model.set_weights(model.get_weights())
                    steps_to_update_target_model = 0
                break
        epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay * episode)
        epsilons.append(epsilon)
        print("epsilon " + str(epsilon))

    plt.xlabel('episode')
    plt.ylabel('Steps')
    xs = range(train_episodes)
    plt.plot(xs, total_steps, 'blue')
    plt.figure(figsize=(10, 10))

    plt.show()
    plt.savefig("steps_" + str(try_number) + ".png")
    plt.close()

    plt.xlabel('episode')
    plt.ylabel('Apples Eaten')
    xs = range(train_episodes)

    plt.plot(xs, total_rewards, 'blue')
    plt.figure(figsize=(10, 10))
    plt.show()
    plt.savefig("score_" + str(try_number) + ".png")
    plt.close()


    return model.get_weights()


def test(try_number, weights):
    test_episodes = 10
    if try_number == 1:
        width = 14
        height = 14
        border = 10
        env = SnakeGame(14, 14, food_amount=1, border=10)
    elif try_number == 2:
        width = 14
        height = 14
        border = 10
        env = SnakeGame(14, 14, food_amount=4, border=10)
    elif try_number == 3:
        width = 30
        height = 30
        border = 1
        env = SnakeGame(30, 30, food_amount=4, border=1)
    else:
        width = 30
        height = 30
        border = 1
        env = SnakeGame(30, 30, border=1)

    model = agent((width + border * 2, height + border * 2, 3), 3)
    model.set_weights(weights)

    total_steps = []
    total_score = []
    for episode in range(test_episodes):
        i = 0
        score = 0
        observation, reward, done, info = env.reset()
        while not done and i < 500:
            i += 1
            reshaped = observation.reshape([1, observation.shape[0], observation.shape[1], observation.shape[2]])
            predicted = model.predict(reshaped).flatten()
            action = np.argmax(predicted)
            aux = action - 1
            new_observation, reward, done, info = env.step(aux)
            plot_board("episode_" + str(episode) + "_step_" + str(i) + ".png", new_observation, aux)
            print("episode_" + str(episode) + "_step_" + str(i))
            observation = new_observation
            if reward >= 1:
                score += 1
            if done:
                total_steps.append(i)
                i = 0
                score = 0

        total_score.append(score)

    plt.xlabel('episode')
    plt.ylabel('Steps')
    xs = range(test_episodes)
    plt.plot(xs, total_steps, 'blue')
    plt.figure(figsize=(10, 10))

    plt.show()
    plt.savefig("steps_test.png")
    plt.close()

    plt.xlabel('episode')
    plt.ylabel('Apples Eaten')
    xs = range(test_episodes)

    plt.plot(xs, total_score, 'blue')
    plt.figure(figsize=(10, 10))
    plt.show()
    plt.savefig("score_test.png")
    plt.close()


first_weights = main(1)
test(1, first_weights)
# second_weights = main(2, first_weights)
# test(2, second_weights)
# third_weights = main(3, second_weights)
# test(3, third_weights)
# fourth_weights = main(4, third_weights)
# test(4, fourth_weights)
