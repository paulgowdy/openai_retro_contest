from retro_contest.local import make

def main():

    env = make(game='SonicTheHedgehog-Genesis', state='LabyrinthZone.Act1')
    obs = env.reset()

    i = 0

    while i < 900:

        action = env.action_space.sample()

        action[7] = 1

        #print(action)

        obs, rew, done, info = env.step(action)

        #print(obs.shape)

        env.render()

        if done:

            obs = env.reset()

        i += 1


if __name__ == '__main__':

    main()
