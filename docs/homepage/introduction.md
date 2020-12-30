---
sort: 1
---

# 简介&安装

## 简介

StewRL框架是为了便于初学者理解强化学习算法的一个模块化强化学习框架。

## 安装

1. 下载代码

   ``` git
   git clone https:/github.com/DouPiChen/StewRL.git
   ```

   

2. 运行代码

   ```python
   python example.py
   ```


## 自定义gym环境配置

1. 进入env-gym/env_gym/envs中添加对应的python代码，如maze_env.py

2. 编辑env-gym/env_gym/envs/\__init__.py将该python文件加入到对应的包中，如下：

   ```python
   from env_gym.envs.maze_env import MazeEnv
   ```
   
3. 编辑env-gym/env_gym/\__init__.py将该python文件注册到gym环境中，如下：
  
  ```python
  from gym.envs.registration import register
  
  register(
           id="maze-v0",
           entry_point="env_gym.envs:MazeEnv",
          )
  ```
  
4. 进入env-gym/文件夹本地安装gym环境，如下

   ```pip
   pip install -e .
   ```

   

   


   

