from retro_contest.local import make


def main():
    env = make(game='SonicTheHedgehog-Genesis', state='LabyrinthZone.Act1')
    obs = env.reset()
    for i in range(100):

        a = env.action_space.sample()

        print(a)



        obs, rew, done, info = env.step(a)
        env.render()
        if done:
            obs = env.reset()


if __name__ == '__main__':
    main()
