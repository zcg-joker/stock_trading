import os
import pandas as pd
import numpy as np

def analyze_transactions(folder_path='results/transactions'):
   """分析交易记录,计算关键指标"""
   results = []
   
   # 遍历transactions文件夹下的所有csv文件
   for filename in os.listdir(folder_path):
       if filename.endswith('.csv'):
           # 获取股票代码
           ticker = filename.split('_')[0]
           
           # 读取csv文件
           df = pd.read_csv(os.path.join(folder_path, filename))
           
           # 计算指标
           # 1. 总收益 = 最终余额 - 初始资金
           initial_money = 10000  # 初始资金
           final_balance = df['total_balance'].iloc[-1]
           total_gains = final_balance - initial_money
           
           # 2. 收益率
           returns = (final_balance - initial_money) / initial_money * 100
           
           # 3. 交易次数
           buy_count = len(df[df['operate'] == 'buy'])
           sell_count = len(df[df['operate'] == 'sell'])
           total_trades = buy_count + sell_count
           
           # 4. 胜率 - 盈利交易次数/总交易次数
           profitable_trades = len(df[df['investment'] > 0])
           win_rate = profitable_trades / sell_count * 100 if sell_count > 0 else 0
           
           # 5. 最大回撤
           # 计算累计资金曲线
           balance_series = df['total_balance']
           # 计算滚动最大值
           rolling_max = balance_series.expanding().max()
           # 计算回撤
           drawdown = (rolling_max - balance_series) / rolling_max * 100
           max_drawdown = drawdown.max()
           
           # 保存结果
           results.append({
               'Stock': ticker,
               'Total Gains ($)': round(total_gains, 2),
               'Returns (%)': round(returns, 2),
               'Total Trades': total_trades,
               'Win Rate (%)': round(win_rate, 2),
               'Max Drawdown (%)': round(max_drawdown, 2)
           })
   
   # 转换为DataFrame并保存
   results_df = pd.DataFrame(results)
   results_df.to_csv('results/output/prediction_metrics.csv', index=False)
   
   # 打印汇总统计
   print("\nTrading Performance Summary:")
   print("="*50)
   print(results_df.describe())
   print("\nDetailed Results by Stock:")
   print("="*50)
   print(results_df)
   
   return results_df

if __name__ == "__main__":
   # 运行分析
   results = analyze_transactions()