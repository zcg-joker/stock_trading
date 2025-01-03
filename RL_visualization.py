import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, List
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class RLVisualizer:
    """强化学习实验结果可视化工具"""
    
    def __init__(self, save_dir: str):
        self.save_dir = save_dir
        # 创建总结果目录
        self.summary_dir = os.path.join(save_dir, 'summary')
        os.makedirs(self.summary_dir, exist_ok=True)
        
        # 设置绘图风格
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def _get_stock_dir(self, ticker: str) -> str:
        """获取股票特定的结果目录"""
        stock_dir = os.path.join(self.save_dir, ticker)
        os.makedirs(stock_dir, exist_ok=True)
        return stock_dir
    
    def _get_agent_dir(self, ticker: str, agent_name: str) -> str:
        """获取特定算法的结果目录"""
        agent_dir = os.path.join(self._get_stock_dir(ticker), agent_name)
        os.makedirs(agent_dir, exist_ok=True)
        return agent_dir
    
    def plot_training_curves(self, rewards_dict: Dict[str, List[float]], ticker: str):
        """绘制训练曲线对比图"""
        # 为每个算法分别绘制训练曲线
        for agent_name, rewards in rewards_dict.items():
            plt.figure(figsize=(12, 6))
            
            # 根据数据长度动态调整窗口大小
            window_size = min(10, max(2, len(rewards) // 10))
            
            # 绘制原始奖励
            plt.plot(rewards, alpha=0.3, label='Raw Reward', linewidth=1, color='gray')
            
            # 计算并绘制平滑奖励
            try:
                rewards_smooth = pd.Series(rewards).rolling(window=window_size, min_periods=1).mean()
                plt.plot(rewards_smooth, label=f'Smoothed Reward (window={window_size})', 
                        linewidth=2, color='blue')
            except Exception as e:
                print(f"Warning: Failed to compute smoothed rewards: {e}")
            
            plt.xlabel('Episode')
            plt.ylabel('Total Reward')
            plt.title(f'{ticker} - {agent_name} Training Process')
            plt.legend(loc='best', fontsize=10, frameon=True, shadow=True)
            plt.grid(True, linestyle='--', alpha=0.7)
            
            # 添加说明文本
            plt.text(0.02, 0.98, 
                    f'Max Reward: {max(rewards):.2f}\n'
                    f'Final Reward: {np.mean(rewards[-min(10, len(rewards)):]):.2f}\n'
                    f'Average Reward: {np.mean(rewards):.2f}',
                    transform=plt.gca().transAxes,
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'),
                    verticalalignment='top',
                    fontsize=10)
            
            # 保存到算法特定目录
            agent_dir = self._get_agent_dir(ticker, agent_name)
            plt.savefig(os.path.join(agent_dir, 'training_rewards.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
        
        # 绘制所有算法对比图
        plt.figure(figsize=(14, 7))
        for agent_name, rewards in rewards_dict.items():
            # 动态调整窗口大小
            window_size = min(10, max(2, len(rewards) // 10))
            try:
                rewards_smooth = pd.Series(rewards).rolling(window=window_size, min_periods=1).mean()
                plt.plot(rewards_smooth, label=f'{agent_name}', linewidth=2)
            except Exception as e:
                print(f"Warning: Failed to compute smoothed rewards for comparison: {e}")
                plt.plot(rewards, label=f'{agent_name} (raw)', linewidth=2)
        
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.title(f'Training Results Comparison - {ticker}')
        plt.legend(loc='best', fontsize=10, frameon=True, shadow=True)
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # 保存到股票特定目录
        plt.savefig(os.path.join(self._get_stock_dir(ticker), 'algorithms_comparison.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_action_distribution(self, actions: List[float], ticker: str, agent_name: str):
        """绘制动作分布图"""
        plt.figure(figsize=(12, 6))
        
        # 计算动作统计信息
        buy_actions = [a for a in actions if a >= 0.1]
        sell_actions = [a for a in actions if a <= -0.1]
        hold_actions = [a for a in actions if -0.1 < a < 0.1]
        
        # 绘制直方图
        plt.hist(actions, bins=50, density=True, alpha=0.7, label='Action Distribution')
        plt.axvline(x=0, color='r', linestyle='--', alpha=0.5, label='Holding the boundary line')
        plt.legend(loc='upper right', fontsize=10, frameon=True, shadow=True)  # 将label置于图的右上角
        
        plt.xlabel('Action Value')
        plt.ylabel('Density')
        plt.title(f'{ticker} - {agent_name} Action Distribution')
        plt.legend(loc='best', fontsize=10, frameon=True, shadow=True)
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # 添加统计信息
        plt.text(0.02, 0.98, 
                f'Number of buy operations: {len(buy_actions)}\n'
                f'Number of sell operations: {len(sell_actions)}\n'
                f'Number of hold operations: {len(hold_actions)}',
                transform=plt.gca().transAxes,
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'),
                verticalalignment='top',
                horizontalalignment='left',  # 置于图的左上角
                fontsize=10)
        
        # 保存到算法特定目录
        agent_dir = self._get_agent_dir(ticker, agent_name)
        plt.savefig(os.path.join(agent_dir, 'action_distribution.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_portfolio_performance(self, history: Dict, ticker: str, agent_name: str):
        """绘制投资组合表现"""
        # 创建子图
        fig = plt.figure(figsize=(15, 12))
        gs = fig.add_gridspec(3, 1, height_ratios=[2, 1, 1], hspace=0.3)
        
        # 1. 股价和交易点图
        ax1 = fig.add_subplot(gs[0])
        ax1.plot(history['timestamps'], history['prices'], 
                label='Price', color='blue', linewidth=2)
        
        # 标注买卖点
        if history['buy_timestamps']:
            ax1.scatter(history['buy_timestamps'], history['buy_prices'], 
                       color='red', marker='^', s=100, label='Buy')
        if history['sell_timestamps']:
            ax1.scatter(history['sell_timestamps'], history['sell_prices'], 
                       color='green', marker='v', s=100, label='Sell')
        
        ax1.set_title(f'{ticker} - Stock Price and Trading Decisions', fontsize=12)
        ax1.set_xlabel('Time step')
        ax1.set_ylabel('Price')
        ax1.legend(loc='best', fontsize=10)
        ax1.grid(True, linestyle='--', alpha=0.7)
        
        # 2. 投资组合价值图
        ax2 = fig.add_subplot(gs[1])
        ax2.plot(history['timestamps'], history['portfolio_values'], 
                label='Portfolio Value', color='purple', linewidth=2)
        ax2.set_xlabel('Time step')
        ax2.set_ylabel('Value')
        ax2.legend(loc='best', fontsize=10)
        ax2.grid(True, linestyle='--', alpha=0.7)
        
        # 3. 持仓量图
        ax3 = fig.add_subplot(gs[2])
        ax3.plot(history['timestamps'], history['positions'], 
                label='Number of holdings', color='orange', linewidth=2)
        ax3.set_xlabel('Time step')
        ax3.set_ylabel('Number of holdings')
        ax3.legend(loc='best', fontsize=10)
        ax3.grid(True, linestyle='--', alpha=0.7)
        
        # 添加总体统计信息
        initial_value = history['portfolio_values'][0]
        final_value = history['portfolio_values'][-1]
        returns = (final_value - initial_value) / initial_value * 100
        
        info_text = (f'Initial capital: {initial_value:.2f}\n'
                    f'Final capital: {final_value:.2f}\n'
                    f'Return: {returns:.2f}%\n'
                    f'Number of trades: {len(history["buy_timestamps"]) + len(history["sell_timestamps"])}')
        
        plt.figtext(0.02, 0.02, info_text, fontsize=10,
                   bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))
        
        # 保存图像
        agent_dir = self._get_agent_dir(ticker, agent_name)
        plt.savefig(os.path.join(agent_dir, 'portfolio_performance.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_summary_results(self, all_results: Dict[str, Dict[str, List[float]]]):
        """绘制汇总结果"""
        # 计算性能数据
        performance_data = []
        for ticker, results in all_results.items():
            for agent_name, rewards in results.items():
                final_reward = np.mean(rewards[-10:])
                max_reward = np.max(rewards)
                performance_data.append({
                    'Ticker': ticker,
                    'Algorithm': agent_name,
                    'Final Reward': final_reward,
                    'Max Reward': max_reward
                })
        
        df = pd.DataFrame(performance_data)
        
        # 1. 绘制箱线图
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=df, x='Algorithm', y='Final Reward')
        plt.title('Final Performance Comparison Across Stocks')
        plt.xlabel('Algorithm')
        plt.ylabel('Final Reward')
        plt.xticks(rotation=45)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(self.summary_dir, 'algorithm_performance_boxplot.png'),
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. 绘制热力图
        plt.figure(figsize=(12, 8))
        pivot_df = df.pivot(index='Ticker', columns='Algorithm', values='Final Reward')
        sns.heatmap(pivot_df, annot=True, fmt='.2f', cmap='RdYlGn', 
                   cbar_kws={'label': 'Final Reward'})
        plt.title('Algorithm Performance Heatmap')
        plt.xlabel('Algorithm')
        plt.ylabel('Stock Code')
        plt.tight_layout()
        plt.savefig(os.path.join(self.summary_dir, 'performance_heatmap.png'),
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. 绘制学习效率对比图
        plt.figure(figsize=(14, 7))
        for agent_name in all_results[list(all_results.keys())[0]].keys():
            avg_rewards = np.mean([results[agent_name] 
                                 for results in all_results.values()], axis=0)
            std_rewards = np.std([results[agent_name] 
                                for results in all_results.values()], axis=0)
            
            episodes = range(len(avg_rewards))
            plt.plot(episodes, avg_rewards, label=f'{agent_name}', linewidth=2)
            plt.fill_between(episodes, 
                           avg_rewards - std_rewards, 
                           avg_rewards + std_rewards, 
                           alpha=0.2)
        
        plt.xlabel('Episode')
        plt.ylabel('Average Reward')
        plt.title('Algorithm Learning Efficiency Comparison')
        plt.legend(loc='best', fontsize=10, frameon=True, shadow=True)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig(os.path.join(self.summary_dir, 'learning_efficiency.png'),
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_summary_report(self, all_results: Dict[str, Dict[str, List[float]]]):
        """创建实验总结报告"""
        report_data = []
        
        for ticker, results in all_results.items():
            for agent_name, rewards in results.items():
                final_reward = np.mean(rewards[-10:])
                max_reward = np.max(rewards)
                stability = np.std(rewards[-50:])
                threshold = 0.8 * final_reward
                learning_speed = np.where(np.array(rewards) >= threshold)[0][0]
                
                report_data.append({
                    'Stock': ticker,
                    'Algorithm': agent_name,
                    'Final Reward': final_reward,
                    'Max Reward': max_reward,
                    'Stability': stability,
                    'Learning Speed': learning_speed
                })
        
        # 创建DataFrame
        report = pd.DataFrame(report_data)
        
        # 保存详细报告
        report.to_csv(os.path.join(self.summary_dir, 'experiment_summary.csv'), index=False)
        
        # 创建性能排名
        rankings = report.groupby('Algorithm').agg({
            'Final Reward': 'mean',
            'Max Reward': 'mean',
            'Stability': 'mean',
            'Learning Speed': 'mean'
        }).round(2)
        
        rankings.to_csv(os.path.join(self.summary_dir, 'algorithm_rankings.csv'))
        
        return report, rankings
    
    def plot_reward_components(self, history: Dict, ticker: str, agent_name: str):
        """
        绘制奖励函数的各个组成部分
        
        参数:
            history: 包含训练历史的字典
            ticker: 股票代码
            agent_name: 算法名称
        """
        # 创建子图
        fig = plt.figure(figsize=(15, 12))
        gs = fig.add_gridspec(3, 2, height_ratios=[2, 1, 1], hspace=0.3, wspace=0.3)
        
        # 1. 基础收益率
        ax1 = fig.add_subplot(gs[0, 0])
        returns = pd.Series(history['returns']).rolling(window=10).mean()
        ax1.plot(returns, color='blue', linewidth=2)
        ax1.set_title('Rolling Returns (10-day MA)')
        ax1.set_xlabel('Time step')
        ax1.set_ylabel('Return')
        ax1.grid(True, linestyle='--', alpha=0.7)
        
        # 2. 夏普比率
        ax2 = fig.add_subplot(gs[0, 1])
        sharpe_ratios = pd.Series(history['sharpe_ratios']).rolling(window=10).mean()
        ax2.plot(sharpe_ratios, color='green', linewidth=2)
        ax2.set_title('Rolling Sharpe Ratio (10-day MA)')
        ax2.set_xlabel('Time step')
        ax2.set_ylabel('Sharpe Ratio')
        ax2.grid(True, linestyle='--', alpha=0.7)
        
        # 3. 最大回撤
        ax3 = fig.add_subplot(gs[1, 0])
        drawdowns = pd.Series(history['drawdowns']).rolling(window=10).mean()
        ax3.plot(drawdowns, color='red', linewidth=2)
        ax3.set_title('Rolling Maximum Drawdown (10-day MA)')
        ax3.set_xlabel('Time step')
        ax3.set_ylabel('Drawdown')
        ax3.grid(True, linestyle='--', alpha=0.7)
        
        # 4. 趋势一致性
        ax4 = fig.add_subplot(gs[1, 1])
        trend_alignments = pd.Series(history['trend_alignments']).rolling(window=10).mean()
        ax4.plot(trend_alignments, color='purple', linewidth=2)
        ax4.set_title('Trend Alignment Score (10-day MA)')
        ax4.set_xlabel('Time step')
        ax4.set_ylabel('Alignment Score')
        ax4.grid(True, linestyle='--', alpha=0.7)
        
        # 5. 综合奖励
        ax5 = fig.add_subplot(gs[2, :])
        total_rewards = pd.Series(history['total_rewards']).rolling(window=10).mean()
        ax5.plot(total_rewards, color='orange', linewidth=2)
        ax5.set_title('Total Reward (10-day MA)')
        ax5.set_xlabel('Time step')
        ax5.set_ylabel('Reward')
        ax5.grid(True, linestyle='--', alpha=0.7)
        
        # 添加统计信息
        stats_text = (
            f'Average Return: {np.mean(history["returns"]):.4f}\n'
            f'Final Sharpe Ratio: {history["sharpe_ratios"][-1]:.4f}\n'
            f'Maximum Drawdown: {np.min(history["drawdowns"]):.4f}\n'
            f'Average Trend Alignment: {np.mean(history["trend_alignments"]):.4f}\n'
            f'Average Total Reward: {np.mean(history["total_rewards"]):.4f}'
        )
        plt.figtext(0.02, 0.02, stats_text, fontsize=10,
                   bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))
        
        # 保存图像
        agent_dir = self._get_agent_dir(ticker, agent_name)
        plt.savefig(os.path.join(agent_dir, 'reward_components.png'),
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_trading_metrics(self, history: Dict, ticker: str, agent_name: str):
        """
        绘制交易指标分析图
        """
        fig = plt.figure(figsize=(15, 12))
        gs = fig.add_gridspec(3, 2, height_ratios=[2, 1, 1], hspace=0.3, wspace=0.3)
        
        # 1. 累积收益曲线
        ax1 = fig.add_subplot(gs[0, :])
        portfolio_values = np.array(history['portfolio_values'])
        returns = (portfolio_values - portfolio_values[0]) / portfolio_values[0] * 100
        ax1.plot(returns, color='blue', linewidth=2, label='Portfolio Returns (%)')
        ax1.set_title('Cumulative Returns')
        ax1.set_xlabel('Time step')
        ax1.set_ylabel('Return (%)')
        ax1.grid(True, linestyle='--', alpha=0.7)
        ax1.legend()
        
        # 2. 交易频率分析
        ax2 = fig.add_subplot(gs[1, 0])
        # 分别获取买入和卖出的时间点
        buy_times = [trade[1] for trade in history['trades'] if trade[0] == 'buy']
        sell_times = [trade[1] for trade in history['trades'] if trade[0] == 'sell']
        
        if buy_times:  # 如果有买入交易
            ax2.hist(buy_times, bins=30, color='green', alpha=0.6, label='Buy')
        if sell_times:  # 如果有卖出交易
            ax2.hist(sell_times, bins=30, color='red', alpha=0.6, label='Sell')
            
        ax2.set_title('Trading Frequency Analysis')
        ax2.set_xlabel('Time step')
        ax2.set_ylabel('Number of Trades')
        ax2.legend()
        ax2.grid(True, linestyle='--', alpha=0.7)
        
        # 3. 持仓量变化
        ax3 = fig.add_subplot(gs[1, 1])
        ax3.plot(history['positions'], color='orange', linewidth=2)
        ax3.set_title('Position Size Changes')
        ax3.set_xlabel('Time step')
        ax3.set_ylabel('Number of Shares')
        ax3.grid(True, linestyle='--', alpha=0.7)
        
        # 4. 资金使用率
        ax4 = fig.add_subplot(gs[2, 0])
        money_utilization = [pos * price / (pos * price + money) * 100 
                           for pos, price, money in zip(history['positions'],
                                                      history['prices'],
                                                      history['portfolio_values'])]
        ax4.plot(money_utilization, color='purple', linewidth=2)
        ax4.set_title('Capital Utilization Rate')
        ax4.set_xlabel('Time step')
        ax4.set_ylabel('Utilization (%)')
        ax4.grid(True, linestyle='--', alpha=0.7)
        
        # 5. 收益分布
        ax5 = fig.add_subplot(gs[2, 1])
        daily_returns = np.diff(portfolio_values) / portfolio_values[:-1]
        ax5.hist(daily_returns, bins=50, color='green', alpha=0.6)
        ax5.axvline(x=0, color='r', linestyle='--', alpha=0.5)
        ax5.set_title('Daily Returns Distribution')
        ax5.set_xlabel('Return')
        ax5.set_ylabel('Frequency')
        ax5.grid(True, linestyle='--', alpha=0.7)
        
        # 添加统计信息
        stats_text = (
            f'Total Return: {returns[-1]:.2f}%\n'
            f'Number of Trades: {len(history["trades"])}\n'
            f'Buy Trades: {len(buy_times)}\n'
            f'Sell Trades: {len(sell_times)}\n'
            f'Average Position Size: {np.mean(history["positions"]):.0f}\n'
            f'Average Capital Utilization: {np.mean(money_utilization):.2f}%\n'
            f'Daily Return Volatility: {np.std(daily_returns) * 100:.2f}%'
        )
        plt.figtext(0.02, 0.02, stats_text, fontsize=10,
                   bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))
        
        # 保存图像
        agent_dir = self._get_agent_dir(ticker, agent_name)
        plt.savefig(os.path.join(agent_dir, 'trading_metrics.png'),
                   dpi=300, bbox_inches='tight')
        plt.close()