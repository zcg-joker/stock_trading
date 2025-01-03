import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import os
import matplotlib.pyplot as plt
from RL_visualization import RLVisualizer
import logging
from datetime import datetime
import json
from tqdm import tqdm

class TradingEnv:
    """
    交易环境类
    
    特点:
    1. 动作空间: 连续值 [-1, 1]
       - (-1, -0.1]: 卖出操作，比例为|action|
       - (-0.1, 0.1): 持有不动
       - [0.1, 1]: 买入操作，比例为action
    2. 状态空间: 包含价格趋势、持仓量、现金等信息
    3. 奖励设计: 
       - 长期收益：累积收益率和夏普比率
       - 风险控制：波动率惩罚和回撤惩罚
       - 交易成本：考虑交易频率和规模
       - 市场适应：考虑市场趋势
    """
    def __init__(self, prices, initial_money=100000, transaction_fee_percent=0.001,
                 reward_scaling=1.0, window_size=20):
        self.prices = prices
        self.initial_money = initial_money
        self.transaction_fee_percent = transaction_fee_percent
        self.reward_scaling = reward_scaling
        self.window_size = window_size
        self.reset()
        
    def reset(self):
        self.t = 0
        self.money = self.initial_money
        self.shares = 0
        self.done = False
        self.portfolio_value = self.money
        self.trades = []  # 记录交易历史
        self.portfolio_values = [self.portfolio_value]  # 记录投资组合价值历史
        self.returns = []  # 记录收益率历史
        return self._get_state()
    
    def _calculate_sharpe_ratio(self):
        """计算夏普比率"""
        if len(self.returns) < 2:
            return 0
        
        returns_array = np.array(self.returns)
        return (np.mean(returns_array) / (np.std(returns_array) + 1e-6)) * np.sqrt(252)  # 年化
    
    def _calculate_max_drawdown(self):
        """计算最大回撤"""
        portfolio_values = np.array(self.portfolio_values)
        peak = np.maximum.accumulate(portfolio_values)
        drawdown = (peak - portfolio_values) / peak
        return np.max(drawdown)
    
    def _calculate_trend_alignment(self, action):
        """计算行动与市场趋势的一致性"""
        if self.t < self.window_size:
            return 0
        
        # 计算短期和长期趋势
        short_window = 5
        long_window = self.window_size
        
        short_trend = (self.prices[self.t] / self.prices[self.t - short_window + 1] - 1)
        long_trend = (self.prices[self.t] / self.prices[self.t - long_window + 1] - 1)
        
        # 判断趋势
        if short_trend > 0 and long_trend > 0:  # 上升趋势
            return 1 if action > 0 else -0.5
        elif short_trend < 0 and long_trend < 0:  # 下降趋势
            return 1 if action < 0 else -0.5
        return 0  # 趋势不明显
    
    def _calculate_reward(self, old_portfolio_value, new_portfolio_value, action):
        """
        计算综合奖励
        
        组成部分：
        1. 基础收益奖励：投资组合价值变化
        2. 长期表现奖励：夏普比率
        3. 风险控制惩罚：最大回撤
        4. 交易成本惩罚：基于交易频率和规模
        5. 趋势一致性奖励：行动与市场趋势的匹配度
        """
        # 1. 计算基础收益率
        returns = (new_portfolio_value - old_portfolio_value) / old_portfolio_value
        self.returns.append(returns)
        self.portfolio_values.append(new_portfolio_value)
        
        # 2. 计算夏普比率奖励
        sharpe_ratio = self._calculate_sharpe_ratio()
        sharpe_reward = np.clip(sharpe_ratio, -1, 1) * 0.1
        
        # 3. 计算最大回撤惩罚
        max_drawdown = self._calculate_max_drawdown()
        drawdown_penalty = -max_drawdown * 0.1
        
        # 4. 计算交易成本惩罚
        trade_penalty = 0
        if len(self.trades) >= 2:
            if self.trades[-1][0] != self.trades[-2][0]:  # 如果发生频繁交易
                trade_penalty = -0.001
        
        # 5. 计算趋势一致性奖励
        trend_reward = self._calculate_trend_alignment(action) * 0.1
        
        # 综合奖励
        total_reward = (
            returns * 0.5 +  # 基础收益占50%
            sharpe_reward * 0.2 +  # 夏普比率占20%
            drawdown_penalty * 0.1 +  # 回撤惩罚占10%
            trade_penalty * 0.1 +  # 交易成本惩罚占10%
            trend_reward * 0.1  # 趋势一致性占10%
        )
        
        return total_reward * self.reward_scaling
    
    def _get_state(self):
        # 计算过去window_size天的价格变化率
        window_size = 10
        if self.t >= window_size:
            price_history = self.prices[self.t-window_size:self.t]
        else:
            price_history = self.prices[:self.t+1]
            if len(price_history) < window_size:
                price_history = [self.prices[0]] * (window_size-len(price_history)) + list(price_history)
        
        price_changes = np.diff(price_history) / price_history[:-1]
        
        # 归一化持仓量和现金
        normalized_shares = self.shares * self.prices[self.t] / self.initial_money
        normalized_money = self.money / self.initial_money
        
        # 添加更多市场特征
        if len(price_history) >= 2:
            volatility = np.std(price_changes)
            momentum = (price_history[-1] / price_history[0]) - 1
        else:
            volatility = 0
            momentum = 0
        
        state = np.concatenate([
            price_changes,
            [normalized_shares],
            [normalized_money],
            [volatility],
            [momentum]
        ])
        return state
    
    def step(self, action):
        """
        执行交易动作
        action: 范围[-1, 1]的连续值
        - (-1, -0.1]: 卖出操作，比例为|action|
        - (-0.1, 0.1): 持有不动
        - [0.1, 1]: 买入操作，比例为action
        """
        self.done = self.t >= len(self.prices) - 1
        
        current_price = self.prices[self.t]
        old_portfolio_value = self.money + self.shares * current_price
        
        # 计算当前的趋势一致性
        trend_alignment = self._calculate_trend_alignment(action)
        
        if not self.done:
            # 根据action的值确定交易类型和比例
            if action >= 0.1:  # 买入
                buy_percentage = action
                max_shares = (self.money * buy_percentage) // current_price
                if max_shares > 0:
                    transaction_cost = max_shares * current_price * self.transaction_fee_percent
                    total_cost = max_shares * current_price + transaction_cost
                    if total_cost <= self.money:
                        self.shares += max_shares
                        self.money -= total_cost
                        self.trades.append(('buy', self.t, max_shares, current_price))
                        
            elif action <= -0.1:  # 卖出
                sell_percentage = abs(action)
                shares_to_sell = int(self.shares * sell_percentage)
                if shares_to_sell > 0:
                    transaction_cost = shares_to_sell * current_price * self.transaction_fee_percent
                    total_revenue = shares_to_sell * current_price - transaction_cost
                    self.shares -= shares_to_sell
                    self.money += total_revenue
                    self.trades.append(('sell', self.t, shares_to_sell, current_price))
            
            # 更新时间步和投资���合价值
            self.t += 1
            new_portfolio_value = self.money + self.shares * self.prices[self.t]
            
            # 计算基础收益率
            returns = (new_portfolio_value - old_portfolio_value) / old_portfolio_value
            self.returns.append(returns)
            self.portfolio_values.append(new_portfolio_value)
            
            # 计算夏普比率
            sharpe_ratio = self._calculate_sharpe_ratio()
            
            # 计算最大回撤
            max_drawdown = self._calculate_max_drawdown()
            
            # 计算交易成本惩罚
            trade_penalty = 0
            if len(self.trades) >= 2:
                if self.trades[-1][0] != self.trades[-2][0]:  # 如果发生频繁交易
                    trade_penalty = -0.001
            
            # 计算综合奖励
            reward = (
                returns * 0.5 +  # 基础收益占50%
                np.clip(sharpe_ratio, -1, 1) * 0.2 +  # 夏普比率占20%
                -max_drawdown * 0.1 +  # 回撤惩罚占10%
                trade_penalty * 0.1 +  # 交易成本惩罚占10%
                trend_alignment * 0.1  # 趋势一致性占10%
            ) * self.reward_scaling
            
            # 更新投资组合价值
            self.portfolio_value = new_portfolio_value
            
        else:
            # 在最后一步，不执行交易，但仍然计算最终的投资组合价值
            new_portfolio_value = self.money + self.shares * current_price
            returns = (new_portfolio_value - old_portfolio_value) / old_portfolio_value
            self.returns.append(returns)
            self.portfolio_values.append(new_portfolio_value)
            
            sharpe_ratio = self._calculate_sharpe_ratio()
            max_drawdown = self._calculate_max_drawdown()
            
            reward = (
                returns * 0.5 +
                np.clip(sharpe_ratio, -1, 1) * 0.2 +
                -max_drawdown * 0.1 +
                trend_alignment * 0.1
            ) * self.reward_scaling
            
            self.portfolio_value = new_portfolio_value
        
        info = {
            'portfolio_value': self.portfolio_value,
            'shares': self.shares,
            'money': self.money,
            'returns': returns,  # 添加基础收益率
            'sharpe_ratio': sharpe_ratio,  # 添加夏普比率
            'max_drawdown': max_drawdown,  # 添加最大回撤
            'trend_alignment': trend_alignment,  # 添加趋势一致性
            'total_reward': reward  # 添加总奖励
        }
        
        return self._get_state(), reward, self.done, info

class DQN(nn.Module):
    """
    Deep Q-Network模型 - 使用更深的网络结构
    """
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )
        
        # 初始化权重
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        return self.network(x)

class DQNAgent:
    """
    DQN智能体 - 支持连续动作空间
    """
    def __init__(self, state_dim, action_dim=16, learning_rate=1e-4, gamma=0.99,
                 epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995,
                 memory_size=10000, batch_size=32, target_update=10):
        self.action_dim = action_dim  # 将动作空间离散化为16个值
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Q网络
        self.q_network = DQN(state_dim, action_dim).to(self.device)
        self.target_network = DQN(state_dim, action_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.memory = deque(maxlen=memory_size)
        
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.target_update = target_update
        self.update_count = 0
        
        # 创建动作空间映射
        self.actions = np.linspace(-1, 1, action_dim)
        
    def select_action(self, state, training=True):
        if training and random.random() < self.epsilon:
            return random.choice(self.actions)
        
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state)
            action_idx = q_values.argmax().item()
            return self.actions[action_idx]
    
    def update_epsilon(self):
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
    
    def store_transition(self, state, action, reward, next_state, done):
        # 将连续动作转换为离散索引
        action_idx = np.abs(self.actions - action).argmin()
        self.memory.append((state, action_idx, reward, next_state, done))
    
    def train(self):
        if len(self.memory) < self.batch_size:
            return
        
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(np.array(actions)).to(self.device)
        rewards = torch.FloatTensor(np.array(rewards)).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(np.array(dones)).to(self.device)
        
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.target_network(next_states).max(1)[0].detach()
        target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        # 梯度裁剪，防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
        
        self.update_count += 1
        if self.update_count % self.target_update == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

def setup_logger(save_dir, ticker):
    """设置日志记录器"""
    log_dir = os.path.join(save_dir, ticker, 'DQN')
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, 'training.log')
    
    # 创建日志记录器
    logger = logging.getLogger(f'{ticker}_DQN')
    logger.setLevel(logging.INFO)
    
    # 创建文件处理器
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    
    # 创建控制台处理器
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    # 创建格式器
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    # 添加处理器
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger

def process_stock(ticker, save_dir, configs):
    try:
        # 设置保存目录
        save_dir = os.path.join(save_dir, 'rlresults')
        os.makedirs(save_dir, exist_ok=True)
        
        # 设置日志记录器
        logger = setup_logger(save_dir, ticker)
        logger.info(f"开始处理股票 {ticker}")
        logger.info(f"当前配置: {configs}")
        
        # 创建可视化工具
        visualizer = RLVisualizer(save_dir)
        
        # 读取预测数据
        df = pd.read_pickle(f'results/predictions/{ticker}_predictions.pkl')
        logger.info(f"成功加载{ticker}的预测数据")
        prices = df.Prediction.values.tolist()
        
        # 创建环境和智能体
        env = TradingEnv(
            prices, 
            initial_money=configs['initial_money'],
            transaction_fee_percent=configs['transaction_fee_percent']
        )
        state_dim = len(env.reset())
        action_dim = configs['action_dim']
        
        agent = DQNAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            learning_rate=configs['learning_rate'],
            gamma=configs['gamma'],
            epsilon_start=configs['epsilon_start'],
            epsilon_end=configs['epsilon_end'],
            epsilon_decay=configs['epsilon_decay'],
            memory_size=configs['memory_size'],
            batch_size=configs['batch_size'],
            target_update=configs['target_update']
        )
        
        logger.info(f"创建交易环境和DQN智能体，状态维度: {state_dim}, 动作维度: {action_dim}")
        
        # 训练数据记录
        all_rewards = []
        all_portfolio_values = []
        all_actions = []
        best_reward = float('-inf')
        best_portfolio_value = 0
        best_actions = []
        best_history = None
        
        # 记录奖励组成部分
        reward_components_history = {
            'returns': [],
            'sharpe_ratios': [],
            'drawdowns': [],
            'trend_alignments': [],
            'total_rewards': []
        }
        
        # 创建训练进度条
        pbar = tqdm(range(configs['episodes']), 
                   desc=f"Training {ticker}",
                   ncols=100,
                   leave=False)
        
        # 训练循环
        for episode in pbar:
            state = env.reset()
            total_reward = 0
            actions = []
            portfolio_values = []
            positions = []
            
            # 记录每个episode的奖励组成部分
            episode_returns = []
            episode_sharpe_ratios = []
            episode_drawdowns = []
            episode_trend_alignments = []
            episode_total_rewards = []
            
            while True:
                action = agent.select_action(state)
                next_state, reward, done, info = env.step(action)
                
                # 记录奖励组成部分
                episode_returns.append(info.get('returns', 0))
                episode_sharpe_ratios.append(info.get('sharpe_ratio', 0))
                episode_drawdowns.append(info.get('max_drawdown', 0))
                episode_trend_alignments.append(info.get('trend_alignment', 0))
                episode_total_rewards.append(reward)
                
                agent.store_transition(state, action, reward, next_state, done)
                agent.train()
                
                total_reward += reward
                actions.append(action)
                portfolio_values.append(info['portfolio_value'])
                positions.append(info['shares'])
                state = next_state
                
                if done:
                    agent.update_epsilon()
                    break
            
            all_rewards.append(total_reward)
            all_portfolio_values.append(portfolio_values[-1])
            all_actions.extend(actions)
            
            # 更新奖励组成部分历史
            reward_components_history['returns'].extend(episode_returns)
            reward_components_history['sharpe_ratios'].extend(episode_sharpe_ratios)
            reward_components_history['drawdowns'].extend(episode_drawdowns)
            reward_components_history['trend_alignments'].extend(episode_trend_alignments)
            reward_components_history['total_rewards'].extend(episode_total_rewards)
            
            if total_reward > best_reward:
                best_reward = total_reward
                best_portfolio_value = info['portfolio_value']
                best_actions = actions
                best_history = {
                    'timestamps': list(range(len(prices))),
                    'prices': prices,
                    'portfolio_values': portfolio_values,
                    'positions': positions,
                    'trades': env.trades,
                    'buy_timestamps': [],
                    'buy_prices': [],
                    'sell_timestamps': [],
                    'sell_prices': [],
                    'returns': episode_returns,
                    'sharpe_ratios': episode_sharpe_ratios,
                    'drawdowns': episode_drawdowns,
                    'trend_alignments': episode_trend_alignments,
                    'total_rewards': episode_total_rewards
                }
            
            # 更新进度条描述
            pbar.set_postfix({
                'Reward': f'{total_reward:.2f}',
                'Portfolio': f'{info["portfolio_value"]:.2f}',
                'Epsilon': f'{agent.epsilon:.2f}'
            })
            
            if (episode + 1) % configs['log_interval'] == 0:
                logger.info(f"Episode {episode + 1}/{configs['episodes']}, "
                          f"Total Reward: {total_reward:.2f}, "
                          f"Portfolio Value: {info['portfolio_value']:.2f}, "
                          f"Epsilon: {agent.epsilon:.2f}")
        
        # 使用最佳动作序列重新执行交易以获取买卖点
        env = TradingEnv(
            prices, 
            initial_money=configs['initial_money'],
            transaction_fee_percent=configs['transaction_fee_percent']
        )
        state = env.reset()
        
        for t, action in enumerate(best_actions):
            if 0.1 <= action <= 1:  # 买入动作
                best_history['buy_timestamps'].append(t)
                best_history['buy_prices'].append(prices[t])
            elif -1 <= action <= -0.1:  # 卖出动作
                best_history['sell_timestamps'].append(t)
                best_history['sell_prices'].append(prices[t])
            _, _, done, _ = env.step(action)
            if done:
                break
        
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
        rewards_dict = {'DQN': all_rewards}
        visualizer.plot_training_curves(rewards_dict, ticker)
        visualizer.plot_action_distribution(all_actions, ticker, 'DQN')
        visualizer.plot_portfolio_performance(best_history, ticker, 'DQN')
        visualizer.plot_reward_components(best_history, ticker, 'DQN')
        visualizer.plot_trading_metrics(best_history, ticker, 'DQN')
        
        # 保存实验结果到JSON文件
        dqn_results = {
            'total_gains': total_gains,
            'investment_return': invest,
            'trades_buy': len(best_history['buy_timestamps']),
            'trades_sell': len(best_history['sell_timestamps']),
            'rewards': all_rewards,
            'reward_components': reward_components_history
        }
        
        results_file = os.path.join(save_dir, ticker, 'DQN', 'results.json')
        with open(results_file, 'w') as f:
            json.dump(dqn_results, f, indent=4)
        
        return {
            'total_gains': total_gains,
            'investment_return': invest,
            'trades_buy': len(best_history['buy_timestamps']),
            'trades_sell': len(best_history['sell_timestamps']),
            'rewards': all_rewards,
            'reward_components': reward_components_history
        }
        
    except Exception as e:
        logger.error(f"处理{ticker}时发生错误: {str(e)}", exc_info=True)
        return None

def main():
    """主函数执行所有股票的交易策略"""
    
    # 配置参数字典
    configs = {
        # 环境参数
        'initial_money': 10000,  # 初始资金
        'transaction_fee_percent': 0.001,  # 交易费率
        
        # 动作空间参数
        'action_dim': 16,  # 动作空间维度
        
        # DQN参数
        'learning_rate': 1e-4,  # 学习率
        'gamma': 0.99,  # 折扣因子
        'epsilon_start': 1.0,  # 初始探索率
        'epsilon_end': 0.01,  # 最终探索率
        'epsilon_decay': 0.995,  # 探索率衰减
        'memory_size': 10000,  # 经验回放池大小
        'batch_size': 256,  # 批量大小
        'target_update': 10,  # 目标网络更新频率
        
        # 训练参数
        'episodes': 100,
        'log_interval': 10,
        
        # 网络结构参数
        'hidden_sizes': [256, 128, 64],  # 隐藏层大小
        'dropout_rate': 0.1,  # Dropout比率
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
    config_file = os.path.join(save_dir, 'rlresults', 'dqn_configs.json')
    os.makedirs(os.path.dirname(config_file), exist_ok=True)
    with open(config_file, 'w') as f:
        json.dump(configs, f, indent=4)
    
    # 记录所有结果
    all_results = {}
    
    # 创建总进度条
    pbar = tqdm(tickers, desc="Processing stocks", ncols=100)
    
    # 处理每只股票
    for ticker in pbar:
        pbar.set_description(f"Processing {ticker}")
        results = process_stock(ticker, save_dir, configs)
        if results is not None:
            all_results[ticker] = {'DQN': results['rewards']}
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