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

class A2CAgent:
    """
    Advantage Actor-Critic (A2C)智能体
    """
    def __init__(self, state_dim, action_dim=1, actor_lr=3e-4, critic_lr=1e-3,
                 gamma=0.99, entropy_coef=0.01, value_loss_coef=0.5,
                 device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.actor = Actor(state_dim, action_dim).to(device)
        self.critic = Critic(state_dim).to(device)
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)
        
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef
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
            entropy = dist.entropy()
            return action.item(), log_prob, entropy
        else:
            # 测试时直接使用均值
            return mu.item(), None, None
    
    def update(self, state, action, reward, next_state, done, log_prob, entropy):
        """更新策略和价值网络"""
        # 转换为tensor
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        next_state = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
        reward = torch.FloatTensor([reward]).to(self.device)
        done = torch.FloatTensor([done]).to(self.device)
        
        # 计算优势函数
        value = self.critic(state)
        next_value = self.critic(next_state)
        target_value = reward + self.gamma * next_value * (1 - done)
        advantage = (target_value - value).detach()
        
        # 计算actor损失
        actor_loss = -(log_prob * advantage) - self.entropy_coef * entropy
        
        # 计算critic损失
        critic_loss = self.value_loss_coef * F.mse_loss(value, target_value.detach())
        
        # 更新actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # 更新critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # 记录训练数据
        self.training_history['actor_losses'].append(actor_loss.item())
        self.training_history['critic_losses'].append(critic_loss.item())
        self.training_history['values'].append(value.item())
        self.training_history['advantages'].append(advantage.item())
        self.training_history['entropies'].append(entropy.item())
        
        return actor_loss.item(), critic_loss.item()

def process_stock(ticker, save_dir, configs):
    """处理单个股票的训练过程"""
    try:
        # 设置日志
        save_dir = os.path.join(save_dir, 'rlresults')  # 确保使用rlresults子目录
        os.makedirs(save_dir, exist_ok=True)
        log_dir = os.path.join(save_dir, ticker, 'A2C')
        os.makedirs(log_dir, exist_ok=True)
        
        # 配置日志记录器
        logger = logging.getLogger(f'{ticker}_A2C')
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
        agent = A2CAgent(
            state_dim=state_dim,
            actor_lr=configs['actor_lr'],
            critic_lr=configs['critic_lr'],
            gamma=configs['gamma'],
            entropy_coef=configs['entropy_coef'],
            value_loss_coef=configs['value_loss_coef']
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
                action, log_prob, entropy = agent.select_action(state)
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
                
                # 更新智能体
                actor_loss, critic_loss = agent.update(
                    state, action, reward, next_state, done, log_prob, entropy
                )
                
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
            
            # 记录每个episode的结果
            all_rewards.append(total_reward)
            all_actions.extend(episode_actions)
            reward_components_history.extend(episode_reward_components)
            
            # 更���最佳结果
            if total_reward > best_reward:
                best_reward = total_reward
                best_portfolio_value = info['portfolio_value']
                best_actions = episode_actions
                best_history = episode_history.copy()
            
            # 更新进度条描述
            pbar.set_postfix({
                'Reward': f'{total_reward:.2f}',
                'Portfolio': f'{info["portfolio_value"]:.2f}',
                'Actor Loss': f'{actor_loss:.4f}'
            })
            
            if (episode + 1) % configs['log_interval'] == 0:
                logger.info(f"Episode {episode + 1}/{configs['episodes']}, "
                          f"Total Reward: {total_reward:.2f}, "
                          f"Portfolio Value: {info['portfolio_value']:.2f}, "
                          f"Actor Loss: {actor_loss:.4f}")
        
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
        # 1. 读取已有的DQN和AC结果（如果存在）
        rewards_dict = {}
        try:
            for algo in ['DQN', 'AC']:
                results_file = os.path.join(save_dir, ticker, algo, 'results.json')
                if os.path.exists(results_file):
                    with open(results_file, 'r') as f:
                        results = json.load(f)
                        rewards_dict[algo] = results['rewards']
                        logger.info(f"成功读取{algo}算法的结果")
        except Exception as e:
            logger.warning(f"无法读取已有结果: {e}")
        
        # 2. 添加A2C的结果
        rewards_dict['A2C'] = all_rewards
        logger.info(f"算法结果统计: {', '.join(rewards_dict.keys())}")
        
        # 3. 绘制对比图
        visualizer.plot_training_curves(rewards_dict, ticker)
        visualizer.plot_action_distribution(all_actions, ticker, 'A2C')
        visualizer.plot_portfolio_performance(best_history, ticker, 'A2C')
        visualizer.plot_reward_components(best_history, ticker, 'A2C')
        visualizer.plot_trading_metrics(best_history, ticker, 'A2C')
        
        # 4. 保存A2C的结果
        a2c_results = {
            'total_gains': total_gains,
            'investment_return': invest,
            'trades_buy': len(best_history['buy_timestamps']),
            'trades_sell': len(best_history['sell_timestamps']),
            'rewards': all_rewards,
            'reward_components': reward_components_history
        }
        
        a2c_results_file = os.path.join(save_dir, ticker, 'A2C', 'results.json')
        with open(a2c_results_file, 'w') as f:
            json.dump(a2c_results, f, indent=4)
        
        return a2c_results
        
    except Exception as e:
        logger.error(f"处理{ticker}时发生错误: {str(e)}", exc_info=True)
        return None

def main():
    """主函数：执行所有股票的A2C算法训练"""
    # 配置参数字典
    configs = {
        # 环境参数
        'initial_money': 10000,
        'transaction_fee_percent': 0.001,
        
        # A2C算法参数
        'actor_lr': 3e-4,
        'critic_lr': 1e-3,
        'gamma': 0.99,
        'entropy_coef': 0.01,
        'value_loss_coef': 0.5,
        
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
    config_file = os.path.join(save_dir, 'rlresults', 'a2c_configs.json')
    os.makedirs(os.path.dirname(config_file), exist_ok=True)
    with open(config_file, 'w') as f:
        json.dump(configs, f, indent=4)
    
    # 记录所有结果
    all_results = {}
    
    # 首先读取已有的DQN和AC结果
    try:
        rlresults_dir = os.path.join(save_dir, 'rlresults')
        for algo in ['DQN', 'AC']:
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
            else:
                print(f"无法读取{ticker}的{algo}算法结果")
                    
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
            # 添加A2C的结果
            all_results[ticker]['A2C'] = results['rewards']
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
    # print("\n可用的算法:", list(next(iter(all_results.values())).keys()))

if __name__ == "__main__":
    main() 