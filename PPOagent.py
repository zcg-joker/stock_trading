import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
import os
import pandas as pd
import logging
from datetime import datetime
import json
from tqdm import tqdm
from RL_visualization import RLVisualizer
from RLagent import TradingEnv  # 重用DQN中的环境

class Actor(nn.Module):
    """
    Actor网络：输出动作的均值和标准差
    """
    def __init__(self, state_dim, action_dim=1, hidden_sizes=[256, 128]):
        super(Actor, self).__init__()
        
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_sizes[0]),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # 动作均值输出层
        self.mu_layer = nn.Sequential(
            nn.Linear(hidden_sizes[1], action_dim),
            nn.Tanh()  # 将输出限制在[-1, 1]范围内
        )
        
        # 动作标准差输出层
        self.sigma_layer = nn.Sequential(
            nn.Linear(hidden_sizes[1], action_dim),
            nn.Softplus()  # 确保标准差为正
        )
        
        # 初始化权重
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight)
            nn.init.constant_(m.bias, 0)
    
    def forward(self, state):
        features = self.net(state)
        mu = self.mu_layer(features)
        sigma = self.sigma_layer(features) + 1e-5  # 添加小值避免数值不稳定
        return mu, sigma

class Critic(nn.Module):
    """
    Critic网络：评估状态价值
    """
    def __init__(self, state_dim, hidden_sizes=[256, 128]):
        super(Critic, self).__init__()
        
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_sizes[0]),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_sizes[1], 1)
        )
        
        # 初始化权重
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight)
            nn.init.constant_(m.bias, 0)
    
    def forward(self, state):
        return self.net(state)

class PPOMemory:
    """
    PPO的经验回放缓冲区
    """
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []
        self.log_probs = []
        self.values = []
        self.advantages = []
        self.returns = []
    
    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.next_states.clear()
        self.dones.clear()
        self.log_probs.clear()
        self.values.clear()
        self.advantages.clear()
        self.returns.clear()
    
    def compute_advantages(self, gamma, gae_lambda):
        """计算广义优势估计(GAE)"""
        values = torch.tensor(self.values, dtype=torch.float32)
        rewards = torch.tensor(self.rewards, dtype=torch.float32)
        # 将布尔类型转换为浮点类型
        dones = torch.tensor(self.dones, dtype=torch.float32)
        
        advantages = []
        gae = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
            gae = delta + gamma * gae_lambda * (1 - dones[t]) * gae
            advantages.insert(0, gae)
        
        self.advantages = advantages
        self.returns = [adv + val for adv, val in zip(advantages, self.values)]

class PPOAgent:
    """
    PPO (Proximal Policy Optimization)智能体
    """
    def __init__(self, state_dim, action_dim=1, actor_lr=3e-4, critic_lr=1e-3,
                 gamma=0.99, gae_lambda=0.95, clip_epsilon=0.2, 
                 value_loss_coef=0.5, entropy_coef=0.01, max_grad_norm=0.5,
                 ppo_epochs=10, batch_size=64,
                 device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.actor = Actor(state_dim, action_dim).to(device)
        self.critic = Critic(state_dim).to(device)
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)
        
        self.memory = PPOMemory()
        
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.ppo_epochs = ppo_epochs
        self.batch_size = batch_size
        self.device = device
        self.action_dim = action_dim
        
        # 用于记录训练数据
        self.training_history = {
            'actor_losses': [],
            'critic_losses': [],
            'values': [],
            'advantages': [],
            'entropies': []
        }
    
    def select_action(self, state, training=True):
        """选择动作"""
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        mu, sigma = self.actor(state)
        
        if training:
            # 从正态分布采样动作
            dist = Normal(mu, sigma)
            action = dist.sample()
            action = torch.clamp(action, -1, 1)  # 限制动作范围
            log_prob = dist.log_prob(action)
            value = self.critic(state)
            return action.item(), log_prob.item(), value.item()
        else:
            # 测试时直接使用均值
            return mu.item(), None, None
    
    def update(self):
        """更新策略和价值网络"""
        # 计算优势和回报
        self.memory.compute_advantages(self.gamma, self.gae_lambda)
        
        # 转换为tensor
        states = torch.FloatTensor(self.memory.states).to(self.device)
        actions = torch.FloatTensor(self.memory.actions).to(self.device)
        old_log_probs = torch.FloatTensor(self.memory.log_probs).to(self.device)
        advantages = torch.FloatTensor(self.memory.advantages).to(self.device)
        returns = torch.FloatTensor(self.memory.returns).to(self.device)
        
        # 标准化优势
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO更新
        for _ in range(self.ppo_epochs):
            # 生成批次数据
            indices = torch.randperm(len(states))
            for start in range(0, len(states), self.batch_size):
                end = start + self.batch_size
                batch_indices = indices[start:end]
                
                # 获取批次数据
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                
                # 计算新的动作分布
                mu, sigma = self.actor(batch_states)
                dist = Normal(mu, sigma)
                new_log_probs = dist.log_prob(batch_actions)
                entropy = dist.entropy().mean()
                
                # 计算比率
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                
                # 计算surrogate损失
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                actor_loss = -torch.min(surr1, surr2).mean()
                
                # 计算value损失
                values = self.critic(batch_states)
                value_loss = F.mse_loss(values, batch_returns.unsqueeze(1))
                
                # 计算总损失
                loss = (actor_loss 
                       + self.value_loss_coef * value_loss 
                       - self.entropy_coef * entropy)
                
                # 更新网络
                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                loss.backward()
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                
                self.actor_optimizer.step()
                self.critic_optimizer.step()
                
                # 记录训练数据
                self.training_history['actor_losses'].append(actor_loss.item())
                self.training_history['critic_losses'].append(value_loss.item())
                self.training_history['values'].append(values.mean().item())
                self.training_history['advantages'].append(batch_advantages.mean().item())
                self.training_history['entropies'].append(entropy.item())
        
        # 清空内存
        self.memory.clear()

def process_stock(ticker, save_dir, configs):
    """处理单个股票的训练过程"""
    try:
        # 设置日志
        save_dir = os.path.join(save_dir, 'rlresults')  # 确保使用rlresults子目录
        os.makedirs(save_dir, exist_ok=True)
        log_dir = os.path.join(save_dir, ticker, 'PPO')
        os.makedirs(log_dir, exist_ok=True)
        
        # 配置日志记录器
        logger = logging.getLogger(f'{ticker}_PPO')
        logger.setLevel(logging.INFO)
        
        # 清除已有的处理器
        if logger.handlers:
            logger.handlers.clear()
        
        # 添加新的处理器
        fh = logging.FileHandler(os.path.join(log_dir, 'training.log'))
        ch = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        logger.addHandler(fh)
        logger.addHandler(ch)
        
        # 创建可视化工具
        visualizer = RLVisualizer(save_dir)
        
        # 加载股票数据
        df = pd.read_pickle(f'results/predictions/{ticker}_predictions.pkl')
        logger.info(f"成功加载{ticker}的预测数据")
        prices = df.Prediction.values.tolist()
        
        # 创建环境和智能体
        env = TradingEnv(
            prices,
            initial_money=configs['initial_money'],
            transaction_fee_percent=configs['transaction_fee_percent']
        )
        # 获取初始状态以确定状态维度
        initial_state = env.reset()
        state_dim = len(initial_state)
        agent = PPOAgent(
            state_dim=state_dim,
            actor_lr=configs['actor_lr'],
            critic_lr=configs['critic_lr'],
            gamma=configs['gamma'],
            gae_lambda=configs['gae_lambda'],
            clip_epsilon=configs['clip_epsilon'],
            value_loss_coef=configs['value_loss_coef'],
            entropy_coef=configs['entropy_coef'],
            ppo_epochs=configs['ppo_epochs'],
            batch_size=configs['batch_size']
        )
        
        # 记录训练过程
        best_reward = float('-inf')
        best_portfolio_value = 0
        best_actions = []
        all_rewards = []
        all_actions = []
        reward_components_history = []
        episode_history = {
            'timestamps': list(range(len(prices))),
            'prices': prices,
            'portfolio_values': [],
            'positions': [],
            'buy_timestamps': [],
            'buy_prices': [],
            'sell_timestamps': [],
            'sell_prices': [],
            'returns': [],
            'sharpe_ratios': [],
            'drawdowns': [],
            'trend_alignments': [],
            'total_rewards': [],
            'trades': []
        }
        best_history = episode_history.copy()
        
        # 训练循环
        pbar = tqdm(range(configs['episodes']), desc=f"Training {ticker}")
        for episode in pbar:
            state = env.reset()
            done = False
            total_reward = 0
            episode_actions = []
            episode_reward_components = []
            episode_history = {
                'timestamps': list(range(len(prices))),
                'prices': prices,
                'portfolio_values': [],
                'positions': [],
                'buy_timestamps': [],
                'buy_prices': [],
                'sell_timestamps': [],
                'sell_prices': [],
                'returns': [],
                'sharpe_ratios': [],
                'drawdowns': [],
                'trend_alignments': [],
                'total_rewards': [],
                'trades': []
            }
            
            current_step = 0
            while not done:
                action, log_prob, value = agent.select_action(state)
                next_state, reward, done, info = env.step(action)
                
                # 记录交易历史
                episode_history['portfolio_values'].append(info['portfolio_value'])
                episode_history['positions'].append(info['shares'])
                episode_history['returns'].append(info['returns'])
                episode_history['sharpe_ratios'].append(info['sharpe_ratio'])
                episode_history['drawdowns'].append(info['max_drawdown'])
                episode_history['trend_alignments'].append(info['trend_alignment'])
                episode_history['total_rewards'].append(info['total_reward'])
                
                # 记录买卖点
                if action >= 0.1:  # 买入
                    episode_history['buy_timestamps'].append(current_step)
                    episode_history['buy_prices'].append(prices[current_step])
                    episode_history['trades'].append(('buy', current_step, info['shares'], prices[current_step]))
                elif action <= -0.1:  # 卖出
                    episode_history['sell_timestamps'].append(current_step)
                    episode_history['sell_prices'].append(prices[current_step])
                    episode_history['trades'].append(('sell', current_step, info['shares'], prices[current_step]))
                
                # 存储经验
                agent.memory.states.append(state)
                agent.memory.actions.append(action)
                agent.memory.rewards.append(reward)
                agent.memory.next_states.append(next_state)
                agent.memory.dones.append(done)
                agent.memory.log_probs.append(log_prob)
                agent.memory.values.append(value)
                
                state = next_state
                total_reward += reward
                episode_actions.append(action)
                current_step += 1
                
                # 添加奖励组成部分
                reward_components = {
                    'returns': info['returns'],
                    'sharpe_ratio': info['sharpe_ratio'],
                    'max_drawdown': info['max_drawdown'],
                    'trend_alignment': info['trend_alignment'],
                    'total_reward': info['total_reward']
                }
                episode_reward_components.append(reward_components)
            
            # 更新策略
            agent.update()
            
            # 记录每个episode的结果
            all_rewards.append(total_reward)
            all_actions.extend(episode_actions)
            reward_components_history.extend(episode_reward_components)
            
            # 更新最佳结果
            if total_reward > best_reward:
                best_reward = total_reward
                best_portfolio_value = info['portfolio_value']
                best_actions = episode_actions
                best_history = episode_history.copy()
            
            # 更新进度条描述
            pbar.set_postfix({
                'Reward': f'{total_reward:.2f}',
                'Portfolio': f'{info["portfolio_value"]:.2f}',
                'Actor Loss': f'{np.mean(agent.training_history["actor_losses"][-10:]):.4f}'
            })
            
            if (episode + 1) % configs['log_interval'] == 0:
                logger.info(f"Episode {episode + 1}/{configs['episodes']}, "
                          f"Total Reward: {total_reward:.2f}, "
                          f"Portfolio Value: {info['portfolio_value']:.2f}, "
                          f"Actor Loss: {np.mean(agent.training_history['actor_losses'][-10:]):.4f}")
        
        # 计算投资回报率
        invest = ((best_portfolio_value - configs['initial_money']) / configs['initial_money']) * 100
        total_gains = best_portfolio_value - configs['initial_money']
        
        # 记录最终结果
        logger.info(f"\n最终结果:"
                   f"\n总收益: {total_gains:.2f}"
                   f"\n投资回报率: {invest:.2f}%"
                   f"\n买入交易次数: {len(best_history['buy_timestamps'])}"
                   f"\n卖出交易次数: {len(best_history['sell_timestamps'])}")
        
        # 可视化结果
        # 1. 读取已有的DQN、AC和A2C结果（如果存在）
        rewards_dict = {}
        try:
            for algo in ['DQN', 'AC', 'A2C']:
                results_file = os.path.join(save_dir, ticker, algo, 'results.json')
                if os.path.exists(results_file):
                    with open(results_file, 'r') as f:
                        results = json.load(f)
                        rewards_dict[algo] = results['rewards']
                        logger.info(f"成功读取{algo}算法的结果")
        except Exception as e:
            logger.warning(f"无法读取已有结果: {e}")
        
        # 2. 添加PPO的结果
        rewards_dict['PPO'] = all_rewards
        logger.info(f"算法结果统计: {', '.join(rewards_dict.keys())}")
        
        # 3. 绘制对比图
        visualizer.plot_training_curves(rewards_dict, ticker)
        visualizer.plot_action_distribution(all_actions, ticker, 'PPO')
        visualizer.plot_portfolio_performance(best_history, ticker, 'PPO')
        visualizer.plot_reward_components(best_history, ticker, 'PPO')
        visualizer.plot_trading_metrics(best_history, ticker, 'PPO')
        
        # 4. 保存PPO的结果
        ppo_results = {
            'total_gains': total_gains,
            'investment_return': invest,
            'trades_buy': len(best_history['buy_timestamps']),
            'trades_sell': len(best_history['sell_timestamps']),
            'rewards': all_rewards,
            'reward_components': reward_components_history
        }
        
        ppo_results_file = os.path.join(save_dir, ticker, 'PPO', 'results.json')
        with open(ppo_results_file, 'w') as f:
            json.dump(ppo_results, f, indent=4)
        
        return ppo_results
        
    except Exception as e:
        logger.error(f"处理{ticker}时发生错误: {str(e)}", exc_info=True)
        return None

def main():
    """主函数：执行所有股票的PPO算法训练"""
    # 配置参数字典
    configs = {
        # 环境参数
        'initial_money': 10000,
        'transaction_fee_percent': 0.001,
        
        # PPO算法参数
        'actor_lr': 3e-4,
        'critic_lr': 1e-3,
        'gamma': 0.99,
        'gae_lambda': 0.95,
        'clip_epsilon': 0.2,
        'value_loss_coef': 0.5,
        'entropy_coef': 0.01,
        'ppo_epochs': 10,
        'batch_size': 256,
    
    # 训练参数
        'episodes': 100,
        'log_interval': 10,
    }
    
    # 股票列表
    tickers = [
      'AAPL', 'MSFT',   # 科技股
            'JPM', 'BAC',             # 金融股
            'JNJ', 'PFE',      # 医药股  
            'XOM', 'CVX',      # 能源股
            'DIS', 'NFLX',    # 消费股
    ]     
    
    # 创建结果目录
    save_dir = 'results'
    os.makedirs(save_dir, exist_ok=True)
    
    # 保存配置信息
    config_file = os.path.join(save_dir, 'rlresults', 'ppo_configs.json')
    os.makedirs(os.path.dirname(config_file), exist_ok=True)
    with open(config_file, 'w') as f:
        json.dump(configs, f, indent=4)
    
    # 记录所有结果
    all_results = {}
    
    # 首先读取已有的DQN、AC和A2C结果
    try:
        rlresults_dir = os.path.join(save_dir, 'rlresults')
        for algo in ['DQN', 'AC', 'A2C']:
            config_file = os.path.join(rlresults_dir, f'{algo.lower()}_configs.json')
            if os.path.exists(config_file):
                for ticker in tickers:
                    results_file = os.path.join(rlresults_dir, ticker, algo, 'results.json')
                    if os.path.exists(results_file):
                        with open(results_file, 'r') as f:
                            results = json.load(f)
                            if ticker not in all_results:
                                all_results[ticker] = {}
                            all_results[ticker][algo] = results['rewards']
                            print(f"成功读取{ticker}的{algo}算法结果")
    except Exception as e:
        print(f"Warning: 无法读取已有结果: {e}")
    
    # 创建总进度条
    pbar = tqdm(tickers, desc="Processing stocks", ncols=100)
    
    # 处理每只股票
    for ticker in pbar:
        pbar.set_description(f"Processing {ticker}")
        results = process_stock(ticker, save_dir, configs)
        if results is not None:
            # 如果股票还没有结果，创建一个新的字典
            if ticker not in all_results:
                all_results[ticker] = {}
            # 添加PPO的结果
            all_results[ticker]['PPO'] = results['rewards']
            pbar.set_postfix({
                'Return': f"{results['investment_return']:.2f}%",
                'Trades': f"{results['trades_buy'] + results['trades_sell']}"
            })
    
    # 创建汇总可视化
    visualizer = RLVisualizer(os.path.join(save_dir, 'rlresults'))
    visualizer.plot_summary_results(all_results)
    report, rankings = visualizer.create_summary_report(all_results)
    
    print("\n算法排名:")
    print(rankings)

if __name__ == "__main__":
    main() 