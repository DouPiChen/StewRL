---
sort: 2
---

# 快速使用

本页面介绍使用StewRL框架编写训练代码example.py的步骤，具体实例参考cartpole_example.py文件。

## 配置gym环境

1. 添加运行算法所需要库文件，代码如下：

   ```python
   import gym
   import env_gym
   from core import *
   ```

   

2. 配置gym环境，代码如下：

   ```python
   env_name = "CartPole-v0"
   env = gym.make(env_name)
   ```

   

3. 计算神经网络输入输出，代码如下：

   ```python
   in_dim = env.observation_space.shape[0]
   if type(env.action_space)==gym.spaces.box.Box: #continous action 
     out_dim = env.action_space.shape[0]
   else:
     out_dim = env.action_space.n
   ```

   

4. 配置环境随机种子，代码如下：

   ```python
   seed = 666
   env.seed(seed)
   np.random.seed(seed)
   torch.manual_seed(seed)
   ```

## 运行强化学习算法

本小节以DQN算法为例：

1. 生成config配置文件，调用core/config/文件夹下的相关类，代码如下：

   ```python
   config = DQNConfig(env_name,in_dim,out_dim)
   ```

   

2. 根据config配置文件选用对应的强化学习算法，具体算法在core/agent/文件夹中相关类中实现，代码如下：

   ```python
   agent = DQNAgent(config.agent)
   ```

   

3. 选用算法使用的Replay Buffer，分为离线训练使用的ReplayBuffer和在线训练的Trajectory，具体实现在core/memory/文件夹中，代码如下：

   ```python
   memory = ReplayBuffer(config.memory)
   ```

   

4. 配置Agent使用的探索策略，不设置时Agent不使用额外的探索策略，具体代码在core/explorer/文件夹中实现，代码如下：

   ```python
   explorer = EpsilonExploration(config.exploration)
   agent.set_explorer(explorer)
   ```

   

5. 配置运行容器Runner，具体代码在core/runner/文件夹中实现，代码如下：

   ```python
   runner = Runner(env,agent,memory,config.runner)
   ```

   

6. 运行算法，通过config.trick配置算法使用的相关trick，代码如下：

   ```python
   runner.train(step,dict(trick=config.trick))
   ```

   
