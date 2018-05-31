import tensorflow as tf

from retro_contest.local import make

from anyrl.algos import DQN
from anyrl.envs import BatchedGymEnv
from anyrl.envs.wrappers import BatchedFrameStack
from anyrl.models import rainbow_models
from anyrl.rollouts import BatchedPlayer, PrioritizedReplayBuffer, NStepPlayer
from anyrl.spaces import gym_space_vectorizer

import gym_remote.exceptions as gre

from sonic_util_6 import AllowBacktracking, make_env, make_env_multi
import time
import random
import numpy as np
import pickle

def eval_net(net, environment):

    environment.reset_start()
    obs = environment.reset_wait()
    #print(obs1[0][0].shape)

    #print(online_net.step(obs1, obs1))

    #for i in range(500):
    n_episodes = 3

    max_timestep = 100

    reward_collector = []

    for i in range(n_episodes):


        reward_sum = 0

        j = 0

        done = False

        while not done and j < max_timestep:

            action = net.step(obs, obs)['actions'][0]

            environment.step_start([action])

            obs, reward, done, _, = environment.step_wait()
            done = done[0]

            #print(action, reward, done)
            reward_sum += reward[0]

            j += 1

        reward_collector.append(reward_sum)

    mean_reward = float(sum(reward_collector))/len(reward_collector)
    reward_std = np.std(np.array(reward_collector))

    return (mean_reward, reward_std)


config = tf.ConfigProto()

config.gpu_options.allow_growth = True

stages = ['SpringYardZone.Act3',
'SpringYardZone.Act2',
'GreenHillZone.Act3',
'GreenHillZone.Act1',
'StarLightZone.Act2',
'StarLightZone.Act1',
'MarbleZone.Act2',
'MarbleZone.Act1',
'MarbleZone.Act3',
'ScrapBrainZone.Act2',
'LabyrinthZone.Act2',
'LabyrinthZone.Act1',
'LabyrinthZone.Act3']


games = ['SonicTheHedgehog-Genesis']

#test_stage = 'SpringYardZone.Act1'
#test_game = 'SonicTheHedgehog-Genesis'

test_envs = [('SonicTheHedgehog-Genesis', 'SpringYardZone.Act1'),
            ('SonicTheHedgehog-Genesis', 'GreenHillZone.Act2'),
            ('SonicTheHedgehog-Genesis', 'StarLightZone.Act3'),
            ('SonicTheHedgehog-Genesis', 'ScrapBrainZone.Act1')]

print('starting tf session')

results_collect = []

with tf.Session(config=config) as sess:

    print('creating env')

    env = AllowBacktracking(make_env(stack=False, scale_rew=False))

    env = BatchedFrameStack(BatchedGymEnv([[env]]), num_images=4, concat=False)

    print('creating agent')

    online_net, target_net = rainbow_models(sess, env.action_space.n,
                              gym_space_vectorizer(env.observation_space),
                              min_val=-200,
                              max_val=200)

    dqn = DQN(online_net, target_net)

    env.close()

    optimize = dqn.optimize(learning_rate=1e-4)

    saver = tf.train.Saver()

    sess.run(tf.global_variables_initializer())

    train_steps = 1000#200000

    for i in range(3):

        stage = random.choice(stages)
        game = random.choice(games)

        print('creating env')
        env = AllowBacktracking(make_env_multi(game, stage, stack=False, scale_rew=False))
        env = BatchedFrameStack(BatchedGymEnv([[env]]), num_images=4, concat=False)

        player = NStepPlayer(BatchedPlayer(env, dqn.online_net), 3)

        print(i, game, stage)
        print('training steps:', train_steps)

        start = time.time()

        dqn.train(num_steps=train_steps, # Make sure an exception arrives before we stop.
                  player=player,
                  replay_buffer=PrioritizedReplayBuffer(500000, 0.5, 0.4, epsilon=0.1),
                  optimize_op=optimize,
                  train_interval=1,
                  target_interval=8192,
                  batch_size=32,
                  min_buffer_size=10000)

        end = time.time()

        print(end - start)

        print('closing env')

        env.close()

        print('eval on test set...')

        # when using multiple games, need to match up games and stages - list of tuples...
        # want mean reward, with error measures

        eval_scores = []

        for test_en in test_envs:

            test_env = AllowBacktracking(make_env_multi(test_en[0], test_en[1], stack=False, scale_rew=False))
            test_env = BatchedFrameStack(BatchedGymEnv([[test_env]]), num_images=4, concat=False)

            eval_scores.append(eval_net(online_net, test_env))

            test_env.close()

        for k in eval_scores:
            print(k)

        print('save nn')

        save_path = saver.save(sess, "saved_models/rainbow6.ckpt")
        print("Model saved in path: %s" % save_path)

        print("")

        results_collect.append(eval_scores)





        # Want to save in the loop, if improvement

    print(results_collect)

    with open('eval_results.p', 'wb') as f:

        pickle.dump(results_collect, f)

    print("")
    print("")

    print('done training')

    print('save nn')

    save_path = saver.save(sess, "saved_models/rainbow6.ckpt")
    print("Model saved in path: %s" % save_path)

    print("")
    print("")
