---
sort: 1
---

# 配置文件类

配置文件类主要用于生成强化学习算法运行过程中所使用的相关参数，具体代码在core/config/文件夹中。

## 功能简介

1. 具体强化学习算法总的配置文件主要为a2c_config.py和dqn_config.py，其中包含中具体算法所需要使用的相关配置类，具体实例如下：

   ```python
   class DQNConfig:
     def __init__(self,name,in_dim,out_dim):
       self.base = BaseConfig(name)
       network_config = QNetworkConfig(in_dim,out_dim)
       self.network_arch = network_config.get_network_arch()
       self.agent = DQNAgentConfig(self.network_arch)  
       self.memory = MemoryConfig(in_dim,out_dim)
       self.runner = RunnerConfig()
       self.exploration = ExplorationConfig()
       self.trick = TrickConfig(name)
   ```

   

2. BaseConfig类包含强化学习算法中共有的基本参数，代码如下：

   ```python
   class BaseConfig:
     def __init__(self,name='CartPole-v0'):
       self.env_name = name
       self.step = 30000
   ```

   

3. BaseNetworkConfig类及其子类主要生成Agent中神经网络所需的dict，示例代码如下：

   ```python
   class BaseNetworkConfig:
     def __init__(self,in_dim,out_dim):
       self.actor_network = [ \
           ("linear",in_dim,128),
           "relu",
           ("linear",128,128),
           "relu",
           ("linear",128,out_dim),
           "softmax"]
   
     def get_network_arch(self):
       return dict(actor=self.actor_network)
   ```

   

4. AgentConfig类及其子类主要保存Agent运行相关的参数，示例代码如下：

   ```python
   class AgentConfig:
     def __init__(self,network_arch):
       self.gamma = 0.9
       self.lr = 1e-3
   
       network_fn = dict()
       for key,value in network_arch.items():
         network_fn.update({key:\
             lambda value=value: Network(value)})
       self.network_fn = network_fn
       
   class DQNAgentConfig(AgentConfig):
     def __init__(self,network_arch):
       super(DQNAgentConfig,self).__init__(network_arch)
       self.target_update = 50    
   ```

   

5. MemoryConfig类主要用于产生算法使用的memory所需相关参数，示例代码如下：

   ```python
   class MemoryConfig:
     def __init__(self,in_dim,out_dim):
       self.memory_size = 10000
       self.batch_size = 32
       self.in_dim = in_dim
       self.out_dim = out_dim
   ```

   

6. RunnerConfig类保存算法运行容器Runner所需要的参数，示例代码如下：

   ```python
   class RunnerConfig:
     def __init__(self):
       self.test_episode = 20
       self.test_num = 10
       self.save_episode = 20
       self.trajectory_done = 5
   ```

   

7. ExplorationConfig类保存算法使用的exploration策略所需要的参数，示例代码如下：

   ```python
   class ExplorationConfig:
     def __init__(self):
       self.epsilon_decay = 1/2000
       self.max_epsilon = 1.0
       self.min_epsilon = 0.1
   ```

   

8. TrickConfig类保存算法使用的trick策略所需要使用的参数，示例代码如下：

   ```python
   class TrickConfig:
     def __init__(self,env_name):
       self.env_name = env_name
       self.reward_shaping_flag = False
   ```

   

9. 保存以及读取配置类生成的json文件，示例代码如下：

   ```python
   def json2class(filename,obj):
     with open(filename,"r") as f:
       js_dict = json.load(f)
       dict2class(js_dict,obj)
   
   def class2json(filename,obj):
     with open(filename,"w") as f:
       json.dump(obj,f,indent=4,cls=ConfigEncoder)
   ```

   