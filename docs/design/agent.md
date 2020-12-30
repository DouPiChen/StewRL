---
sort: 2
---

# 算法实现类

主要包含BaseAgent类及其子类，具体实现相关强化学习算法的核心部分。

## BaseAgent类接口

1. 初始化函数\__init__()，示例代码如下：

   ```python
     def __init__(self,config):
       self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
       self.gamma = config.gamma
       self.actor = None
       self.optimizer = None
       self.explorer = None
   ```

   

2. 动作选择函数select_action()，示例代码如下：

   ```python
     def select_action(self,state):
       state = torch.FloatTensor(state).to(self.device)
       action = self.actor(state)
       infos = dict(state=state)
       return action,infos
   ```

   

3. 模型更新函数update_model()，示例代码如下：

   ```python
     def update_model(self,sample):
       raise NotImplementedError
   ```

   

4. 损失计算函数compute_loss()，示例代码如下：

   ```python
     def compute_loss(self,sample):
       raise NotImplementedError
   ```

   

5. 反向传播函数backpropagate()，示例代码如下：

   ```python
     def backpropagate(self,loss):
       self.optimizer.zero_grad()
       loss.backward()
       self.optimizer.step()
   ```

   

6. Tensor转换函数trans_tensor()，示例代码如下：

   ```python
     def trans_tensor(self,sample):
       raise NotImplementedError
   ```

## DQNAgent类

以DQNAgent类为例，说明如何实现一个具体的Value-Based的强化学习算法。

1. 初始化函数\__init__()需要配置actor网络和target网络，算法使用的optimizer以及target网络的更新间隔，具体代码如下：

   ```python
     def __init__(self,config):
       super(DQNAgent,self).__init__(config)
       self.actor = config.network_fn["actor"]().to(self.device)
       self.target = config.network_fn["actor"]().to(self.device)
       self.target.load_state_dict(self.actor.state_dict())
       self.target.eval()
       self.optimizer = optim.Adam(self.actor.parameters(),lr=config.lr)
       self.target_update = config.target_update
       self.update_cnt = 0
   ```

   

2. 动作选择函数select_action()中使用actor网络计算的最大Q值选取动作，同时使用epsilon-exploration进行探索，具体代码如下：

   ```python
     def select_action(self,state):
       state = torch.FloatTensor(state).to(self.device)
       action_score = self.actor(state)
       action = action_score.argmax(dim=-1)
       if self.explorer!=None:
         action_dim = action_score.size()[-1]
         action = self.explorer.action_for_exploration(
                      dict(action=action,update_cnt=self.update_cnt,
                           action_dim=action_dim))
       return action,None
   ```

   

3. 损失计算函数compute_loss()用于计算算法优化所使用损失，计算公式如下：

   $$L_i(\theta_i^{actor})=E_{(s,a,r,s^n)\sim U(D)}[r+\gamma max_{a^n}Q(s^n,a^n|\theta_i^{target})-Q(s,a|\theta_i^{actor})]$$

   其中$\theta_i^{actor},\theta_i^{target}$分别为actor网络和target网络的参数，$n$表示next。具体代码实现如下：

   ```python
     def compute_loss(self,sample):
       device = self.device
       sample = self.trans_tensor(sample)
   
       q_curr = self.actor(sample["state"]).gather(1,sample["action"])
       q_next = self.target(sample["next_state"]).max(dim=1,keepdim=True)[0].detach()
       mask = 1-sample["done"]
       q_target = (sample["reward"]+self.gamma*q_next*mask).to(device)
       loss = F.smooth_l1_loss(q_curr,q_target,reduction="none")
       return loss
   ```

   

4. 模型更新函数update_model()，包括actor网络的更新以及target网络载入actor网络更新等2个过程，具体代码如下：

   ```python
     def update_model(self,sample):
       loss = self.compute_loss(sample).mean()
       self.backpropagate(loss)
       self.update_cnt += 1
       if self.update_cnt%self.target_update==0:
         self.target_network_update()
       return loss,None
   ```

   

5. Tensor转换函数trans_tensor()将ReplayBuffer类中sample得到的数据转换为actor网络可以处理的Tensor，具体代码如下：

   ```python
     def trans_tensor(self,sample):
       device = self.device
       sample["state"] = torch.FloatTensor(sample["state"]).to(device)
       sample["next_state"] = torch.FloatTensor(sample["next_state"]).to(device)
       sample["action"] = torch.LongTensor(sample["action"].reshape(-1,1)).to(device)
       sample["reward"] = torch.FloatTensor(sample["reward"].reshape(-1,1)).to(device)
       sample["done"] = torch.FloatTensor(sample["done"].reshape(-1,1)).to(device)
       return sample
   ```

## A2CAgent类

以A2CAgent类为例，说明如何实现一个具体的Policy-Based的强化学习算法。

1. 初始化函数\__init__()需要配置actor网络以及critic网络，同时设置算法使用的optimizer以及loss参数，具体代码如下：

   ```python
     def __init__(self,config):
       super(A2CAgent,self).__init__(config)
       self.actor = config.network_fn["actor"]().to(self.device)
       self.critic = config.network_fn["critic"]().to(self.device)
       self.optimizer = optim.Adam([{"params":self.actor.parameters()},
                             {"params":self.critic.parameters()}],lr=config.lr)
       self.actor_loss_coeff = config.actor_loss_coeff
       self.critic_loss_coeff = config.critic_loss_coeff
       self.entropy_coeff = config.entropy_coeff
   ```

   

2. 动作选择函数select_action()，通过Categorical类采样得到相应的action，同时使用critic网络计算相应的value，具体代码如下：

   ```python
     def select_action(self,state):
       prob,add_infos = super().select_action(state)
       state = add_infos["state"]
       dist = Categorical(prob)
       action = dist.sample()
       logprob = dist.log_prob(action).unsqueeze(0)
       entropy = dist.entropy().mean().unsqueeze(0)
       value = self.critic(state)
       infos = dict(logprob=logprob,value=value,entropy=entropy)
       return action,infos
   ```

   

3. 损失计算函数compute_loss()计算相应的损失，具体计算公式如下：

   $$L_i(\theta_i^{actor})=-E_{(s,a,r,s^n)}log\pi(a|\theta_i^{actor})[G(s,a)-V(s|\theta_i^{critic})]$$

   $$L_i(\theta_i^{critic})=\frac{1}{n}\sum[G(s,a)-V(s|\theta_i^{critic})]$$

   具体代码如下：

   ```python
     def compute_loss(self,sample):
       returns = self.compute_return(sample["reward"],sample["done"])
       sample = self.trans_tensor(sample)
       advantage = returns-sample["value"]
       actor_loss = -(sample["logprob"]*advantage.detach()).mean()
       critic_loss = F.smooth_l1_loss(sample["value"],returns).mean()
       entropy_loss = sample["entropy"].mean()
       return actor_loss,critic_loss,entropy_loss
   ```

   

4. 模型更新函数update_model()，使用计算得到的loss进行梯度反向传播，具体代码如下：

   ```python
     def update_model(self,sample):
       actor_loss,critic_loss,entropy_loss = self.compute_loss(sample)
       loss = self.actor_loss_coeff*actor_loss+ \
              self.critic_loss_coeff*critic_loss+ \
              self.entropy_coeff*entropy_loss
       self.backpropagate(loss)
       return loss,None
   ```

   

5. Tensor转换函数trans_tensor()将Trajectory类中得到的数据转换为actor网络以及ciritic网络可以处理的Tensor，具体代码如下：

   ```python
     def trans_tensor(self,sample):
       for key,value in sample.items():
         if len(value)>0:
           if type(value[0])==np.ndarray:
             value = np.array(value)
             value = torch.FloatTensor(value).to(self.device)
           elif type(value[0])==torch.Tensor:
             value = torch.cat(value)
           else:
             value = torch.FloatTensor(value).to(self.device)
           sample[key] = value
       return sample
   ```

   