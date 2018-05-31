#!/usr/bin/env python

"""
Train an agent on Sonic using an open source Rainbow DQN
implementation.
"""

import tensorflow as tf

from retro_contest.local import make

from anyrl.algos import DQN
from anyrl.envs import BatchedGymEnv
from anyrl.envs.wrappers import BatchedFrameStack
from anyrl.models import rainbow_models
from anyrl.rollouts import BatchedPlayer, PrioritizedReplayBuffer, NStepPlayer
from anyrl.spaces import gym_space_vectorizer

import gym_remote.exceptions as gre

from sonic_util import AllowBacktracking, make_env

import time

def main():

    """Run DQN until the environment throws an exception."""

    print('creating env')

    env = AllowBacktracking(make_env(stack=False, scale_rew=False))

    env = BatchedFrameStack(BatchedGymEnv([[env]]), num_images=4, concat=False)

    config = tf.ConfigProto()

    config.gpu_options.allow_growth = True # pylint: disable=E1101

    print('starting tf session')

    with tf.Session(config=config) as sess:

        print('creating agent')

        online_net, target_net = rainbow_models(sess, env.action_space.n,
                                  gym_space_vectorizer(env.observation_space),
                                  min_val=-200,
                                  max_val=200)

        dqn = DQN(online_net, target_net)

        player = NStepPlayer(BatchedPlayer(env, dqn.online_net), 3)

        optimize = dqn.optimize(learning_rate=1e-4)

        saver = tf.train.Saver()

        sess.run(tf.global_variables_initializer())



        train_steps = 5000

        print('training steps:', train_steps)

        for j in range(1):

            print(j)

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

        print('done training')

        print('save nn')

        save_path = saver.save(sess, "saved_models/rainbow5.ckpt")
        print("Model saved in path: %s" % save_path)

        tvars = tf.trainable_variables()
        tvars_vals = sess.run(tvars)

        #for var, val in zip(tvars, tvars_vals):
        #    print(var.name, val[0])

        #print(tvars_vals[0][-5:])

        #print('stepping')

        #obs = env.reset()

        #online_net.step(obs, obs)

        '''
        i = 0
        while i < 100:
            #action = env.action_space.sample()
            #action[7] = 1
            #print(action)
            obs, rew, done, info = env.step(action)
            #print(obs.shape)
            env.render()
            if done:
                obs = env.reset()
            i += 1
        '''


if __name__ == '__main__':

    print('in main')

    try:

        main()

    except gre.GymRemoteError as exc:

        print('exception', exc)
