import parl
from parl import layers
import paddle.fluid as fluid
import copy
import numpy as np
import os
import gym
from parl.utils import logger
from parl.algorithms import DQN
import paddle
from Agent import Agent
from Model import Model
from ReplayMemory import ReplayMemory

from ple.games.flappybird import FlappyBird
from ple.ple import PLE

import os

import pygame.display
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"

#LEARN_FREQ = 10 # 训练频率，不需要每一个step都learn，攒一些新增经验后再learn，提高效率
#MEMORY_SIZE = 2000    # replay memory的大小，越大越占用内存
#MEMORY_WARMUP_SIZE = 1000  # replay_memory 里需要预存一些经验数据，再开启训练
#BATCH_SIZE = 64   # 每次给agent learn的数据数量，从replay memory随机里sample一批数据出来
LEARNING_RATE = 0.001 # 学习率
GAMMA = 0.99 # reward 的衰减因子，一般取 0.9 到 0.999 不等

def remapGameState(obs):
    obs = [
        obs["player_y"],
        obs["player_vel"],
        obs["next_pipe_dist_to_player"],
        obs["next_pipe_top_y"],
        obs["next_pipe_bottom_y"],
        obs["next_next_pipe_dist_to_player"],
        obs["next_next_pipe_top_y"],
        obs["next_next_pipe_bottom_y"]
        ]
    return obs

def remapAction(act):
    if act == 0:
        return None
    if act == 1:
        return 119


# 评估 agent, 跑 5 个episode，总reward求平均
def evaluate(env, agent, render=False):
    eval_reward = []
    for i in range(5):
        env.reset_game()
        obs = remapGameState(env.getGameState())
        episode_reward = 0
        while True:
            action = agent.predict(obs)  # 预测动作，只选最优动作

            reward= env.act(remapAction(action))
            next_obs = remapGameState(env.getGameState())
            done = env.game_over()

            episode_reward += reward
            if done:
                break
        eval_reward.append(episode_reward)
    return np.mean(eval_reward)
 
game = FlappyBird()
env = PLE(game, fps=30, display_screen=True, force_fps=False)
env.reset_game()
print(env.getGameState())

action_dim = 2 #动作
obs_shape = [8] #观察

# 根据parl框架构建agent
model = Model(act_dim=action_dim)
algorithm = DQN(model, act_dim=action_dim, gamma=GAMMA, lr=LEARNING_RATE)
agent = Agent(
    algorithm,
    obs_dim=obs_shape[0],
    act_dim=action_dim,
    e_greed=0.5,  # 有一定概率随机选取动作，探索
    e_greed_decrement=10e-7)  # 随着训练逐步收敛，探索的程度慢慢降低

# 加载缓存模型
save_path = './Model/dqn_model_8250.ckpt'
if os.path.exists(save_path):
    agent.restore(save_path)

while True:  # 训练max_episode个回合，test部分不计算入episode数量
    env.reset_game()
    obs = remapGameState(env.getGameState())
    while True:
        action = agent.predict(obs)  # 预测动作，只选最优动作

        reward= env.act(remapAction(action))
        next_obs = remapGameState(env.getGameState())
        done = env.game_over()

        pygame.display.set_caption("score:%d"%int(env.score()))
        obs = next_obs
        if done:
            break
