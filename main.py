from unityagents import UnityEnvironment
import numpy as np
import torch
import torch.optim as optim
import random
from collections import deque, namedtuple
from model import QNetwork

from agent_dqn import *


LR = 1e-4
BUFFER_SIZE = int(1e5)
BATCH_SIZE = 40
GAMMA = 0.77
TAU = 1e-3
UPDATE_EVERY = 3


env = UnityEnvironment(file_name="Banana.app")

# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

print_every = 100


agent = Agent(37, 4, 0)

def run(agent, num_episodes=2000, eps_decay=0.995, eps_min=0.01, max_t=1000):
    """
    Runs and trains the agent.
    
    Params
    =======
    
    
    """
    epsilon = 1.0
    
    # create the scores list that will store the scores for each episode
    scores = []
    scores_window = deque(maxlen=100)
    for i_episode in range(1, num_episodes+1):
        
        # intialize the episode with a new state
        env_info = env.reset(train_mode=True)[brain_name]   # reset the environment
        state = env_info.vector_observations[0]     #  collect the state from the API
    
        # initialize the variable score that will store the cumulative rewards during the episode
        score = 0
        for t in range(max_t):
            action = agent.act(state, epsilon)
            # send the chosen action to the environment and collect the observation
            env_info = env.step(action)[brain_name]
            next_state = env_info.vector_observations[0]
            reward = env_info.rewards[0]
            done = env_info.local_done[0]
            
            # store the experience and learn if enough new experiences stored
            agent.step(state, action, reward, next_state, done)
            
            state = next_state

            # update the score
            score += reward
            if done:
                break
        scores_window.append(score) 
        scores.append(score)

        
        epsilon = max(epsilon*eps_decay, eps_min)
#         print('\rEpisode {}/{} - Average score : {:.2f} // size_buffer: {}'.format(i_episode, num_episodes, np.mean(scores_window), agent.memory.__len__()), end="")    
        print('\rEpisode {}/{} - Average score : {:.2f} // Epsilon: {}'.format(i_episode, num_episodes, np.mean(scores_window), epsilon), end="")    
        if i_episode % print_every == 0:
            print('\rEpisode {}/{} - Average score: {:.2f}'.format(i_episode, num_episodes, np.mean(scores_window)))
            
#         if (np.mean(scores_window)>13):
#             print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(scores_window)))
#             torch.save(agent.q_network_local.state_dict(),'navigation.pth')
#             break
#     torch.save(agent.q_network_local.state_dict(),'navigation.pth')
    return scores


scores = run(agent)
