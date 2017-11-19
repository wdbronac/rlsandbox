import gym
from agents.simpleDQN import SimpleDQNAgent
from agents.simpleKNN import SimpleKNNAgent

env = gym.make('CartPole-v0')
# env = gym.make('MountainCar-v0')
agent = SimpleKNNAgent(env)
# agent = SimpleDQNAgent(env)


n_episode = 0
while(True):
    state = env.reset()
    # env.render()
    done = False
    length = 0
    while not done:
        action = agent.act(state)
        next_state, reward, done, info = env.step(action)
        if done:
            reward = -1
        # env.render()
        agent.update(state, action, next_state, reward)
        state = next_state
        env.render()
        length += 1
    n_episode += 1
    print("length: {}".format(length))
    print("n° episode: {}".format(n_episode))