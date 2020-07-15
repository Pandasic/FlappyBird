#该文件为继续训练后的修改文件
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

import pygame.display
LEARN_FREQ = 10 # 训练频率，不需要每一个step都learn，攒一些新增经验后再learn，提高效率
MEMORY_SIZE = 2000    # replay memory的大小，越大越占用内存
MEMORY_WARMUP_SIZE = 1000  # replay_memory 里需要预存一些经验数据，再开启训练
BATCH_SIZE = 64   # 每次给agent learn的数据数量，从replay memory随机里sample一批数据出来
LEARNING_RATE = 0.0001 # 学习率
GAMMA = 0.99 # reward 的衰减因子，一般取 0.9 到 0.999 不等

import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"

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

episode = 0
eval_time = 0

maxScore = 0
maxScoreEp = episode

# 训练一个episode
def run_episode(env, agent, rpm):
    global maxScore,maxScoreEp
    total_reward = 0
    env.reset_game()
    obs = remapGameState(env.getGameState())
    step = 0
    while True:
        step += 1
        action = agent.sample(obs)  # 采样动作，所有动作都有概率被尝试到
        reward= env.act(remapAction(action))
        next_obs = remapGameState(env.getGameState())
        done = env.game_over()

        rpm.append((obs, action, reward, next_obs, done))

        # train model
        if (len(rpm) > MEMORY_WARMUP_SIZE) and (step % LEARN_FREQ == 0):
            (batch_obs, batch_action, batch_reward, batch_next_obs,
             batch_done) = rpm.sample(BATCH_SIZE)
            train_loss = agent.learn(batch_obs, batch_action, batch_reward,
                                     batch_next_obs,
                                     batch_done)  # s,a,r,s',done

        pygame.display.set_caption("score:%d,maxScore:%d,EP = %d"%(int(env.score()),maxScore,maxScoreEp))
        if env.score() >maxScore:
            maxScore = env.score()
            maxScoreEp = episode

        total_reward += reward
        obs = next_obs
        if done:
            break
    return total_reward

# 评估 agent, 跑 5 个episode，总reward求平均
def evaluate(env, agent, render=False):
    global maxScore,maxScoreEp
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

            obs = next_obs
            episode_reward += reward
            if done:
                break
            pygame.display.set_caption("score:%d,maxScore:%d,EP = %d"%(int(env.score()),maxScore,maxScoreEp))
            if env.score() >maxScore:
                maxScore = env.score()
                maxScoreEp = episode

        eval_reward.append(episode_reward)
    return np.mean(eval_reward)
 
game = FlappyBird()
env = PLE(game, fps=30, display_screen=True, force_fps=False)
env.reset_game()
print(env.getGameState())

action_dim = 2 #动作
obs_shape = [8] #观察

rpm = ReplayMemory(MEMORY_SIZE)  # DQN的经验回放池

# 根据parl框架构建agent
model = Model(act_dim=action_dim)
algorithm = DQN(model, act_dim=action_dim, gamma=GAMMA, lr=LEARNING_RATE)
agent = Agent(
    algorithm,
    obs_dim=obs_shape[0],
    act_dim=action_dim,
    e_greed=0.5,  # 有一定概率随机选取动作，探索
    #e_greed=0.05,  # 有一定概率随机选取动作，探索
    e_greed_decrement=10e-7)  # 随着训练逐步收敛，探索的程度慢慢降低

# 加载缓存模型
save_path = './Model/dqn_model_temp.ckpt'
if os.path.exists(save_path):
    agent.restore(save_path)

#evaluate(env, agent, render=False) 

# 先往经验池里存一些数据，避免最开始训练的时候样本丰富度不够
while len(rpm) < MEMORY_WARMUP_SIZE:
    run_episode(env, agent, rpm)

max_episode = 50000

# 开始训练


while episode < max_episode:  # 训练max_episode个回合，test部分不计算入episode数量
    # train part
    total_reward = 0
    for i in range(0, 50):
        reward = run_episode(env, agent, rpm)
        total_reward += reward
        #print(reward)
        episode += 1

    # test part
    #eval_reward = total_reward/50

    eval_reward = evaluate(env, agent, render=False)  # render=True 查看显示效果
    logger.info('episode:{}    e_greed:{}   test_reward:{}    total_reward:{}'.format(
        episode, agent.e_greed, eval_reward,total_reward))
    eval_time += 1

    save_path = './Model/dqn_model_temp.ckpt'
    agent.save(save_path)

    if eval_time%1 ==0:
        save_path = './Model/dqn_model_%d.ckpt'%(episode)
        agent.save(save_path)

# 训练结束，保存模型
save_path = './Model/dqn_model.ckpt'
agent.save(save_path)