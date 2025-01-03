import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

def plot_stock_prediction(ticker, test_indices, actual_prices, predicted_prices, metrics, save_dir):
    """
    绘制股票预测结果对比图
    
    参数:
        ticker: 股票代码
        test_indices: 测试集日期索引
        actual_prices: 实际价格
        predicted_prices: 预测价格
        metrics: 包含rmse、mae和accuracy的字典
        save_dir: 图片保存的根目录
    返回:
        str: 保存的图片路径
    """
    plt.figure(figsize=(15, 7))
    plt.plot(test_indices, actual_prices, label='Actual Price', color='blue', linewidth=2, alpha=0.7)
    plt.plot(test_indices, predicted_prices, label='LSTM Prediction', color='red', linewidth=2, linestyle='--', alpha=0.7)
    
    plt.title(f'{ticker} Stock Price Prediction\nRMSE: {metrics["rmse"]:.2f}, MAE: {metrics["mae"]:.2f}')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.text(0.02, 0.95, f'Prediction Accuracy: {metrics["accuracy"]*100:.2f}%', 
             transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    prediction_dir = os.path.join(save_dir, 'pic/predictions')
    os.makedirs(prediction_dir, exist_ok=True)
    save_path = os.path.join(prediction_dir, f'{ticker}_prediction.png')
    plt.savefig(save_path)
    plt.close()
    
    return save_path

def plot_training_loss(ticker, train_losses, val_losses, save_dir):
    """
    绘制训练和验证损失曲线
    
    参数:
        ticker: 股票代码
        train_losses: 训练损失列表
        val_losses: 验证损失列表
        save_dir: 图片保存的根目录
    返回:
        str: 保存的图片路径
    """
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Training and Validation Loss for {ticker}')
    plt.legend()
    plt.grid(True)
    
    loss_dir = os.path.join(save_dir, 'pic/loss')
    os.makedirs(loss_dir, exist_ok=True)
    save_path = os.path.join(loss_dir, f'{ticker}_loss.png')
    plt.savefig(save_path)
    plt.close()
    
    return save_path

def plot_cumulative_earnings(ticker, test_indices, actual_percentages, predict_percentages, save_dir):
    """
    绘制累积收益率曲线
    
    参数:
        ticker: 股票代码
        test_indices: 测试集日期索引
        actual_percentages: 实际收益率列表
        predict_percentages: 预测收益率列表
        save_dir: 图片保存的根目录
    返回:
        str: 保存的图片路径
    """
    cumulative_naive_percentage = np.cumsum(actual_percentages)
    cumulative_lstm_percentage = np.cumsum(
        [a if p > 0 else 0 for p, a in zip(predict_percentages, actual_percentages)]
    )

    plt.figure(figsize=(10, 6))
    plt.plot(test_indices, cumulative_naive_percentage, marker='o', markersize=3, 
             linestyle='-', color='blue', label='Naive Strategy')
    plt.plot(test_indices, cumulative_lstm_percentage, marker='o', markersize=3, 
             linestyle='-', color='orange', label='LSTM Strategy')
    plt.title(f'Cumulative Earnings Percentages for {ticker}')
    plt.xlabel('Date')
    plt.ylabel('Percentage (%)')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    
    earnings_dir = os.path.join(save_dir, 'pic/earnings')
    os.makedirs(earnings_dir, exist_ok=True)
    save_path = os.path.join(earnings_dir, f'{ticker}_cumulative.png')
    plt.savefig(save_path)
    plt.close()
    
    return save_path

def plot_accuracy_comparison(prediction_metrics, save_dir):
    """
    绘制所有股票预测准确度对比图
    
    参数:
        prediction_metrics: 包含每个股票预测指标的字典
        save_dir: 图片保存的根目录
    返回:
        str: 保存的图片路径
    """
    plt.figure(figsize=(15, 6))
    accuracies = [metrics['accuracy'] * 100 for metrics in prediction_metrics.values()]
    bars = plt.bar(prediction_metrics.keys(), accuracies)
    
    # 在每个柱顶端显示值
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval, f'{yval:.2f}', ha='center', va='bottom')
    
    plt.title('Prediction Accuracy Across Stocks')
    plt.xlabel('Stock')
    plt.ylabel('Accuracy (%)')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    prediction_dir = os.path.join(save_dir, 'pic')
    os.makedirs(prediction_dir, exist_ok=True)
    save_path = os.path.join(prediction_dir, 'accuracy_comparison.png')
    plt.savefig(save_path)
    plt.close()
    
    return save_path

def plot_trading_result(ticker, close_prices, states_buy, states_sell, total_gains, invest, save_dir):
    """
    绘制交易结果图表
    
    参数:
        ticker: 股票代码
        close_prices: 收盘价列表
        states_buy: 买入点列表
        states_sell: 卖出点列表
        total_gains: 总收益
        invest: 投资回报率
        save_dir: 保存路径
    返回:
        str: 保存的图片路径
    """
    plt.figure(figsize=(15, 5))
    plt.plot(close_prices, color='r', lw=2.)
    plt.plot(close_prices, '^', markersize=10, color='m', label='buying signal', markevery=states_buy)
    plt.plot(close_prices, 'v', markersize=10, color='k', label='selling signal', markevery=states_sell)
    plt.title(f'{ticker} total gains ${total_gains:.2f}, total investment {invest:.2f}%')
    plt.legend()
    
    # 创建保存目录
    trades_dir = os.path.join(save_dir, 'pic/trades')
    os.makedirs(trades_dir, exist_ok=True)
    
    # 保存图片
    save_path = os.path.join(trades_dir, f'{ticker}_trades.png')
    plt.savefig(save_path)
    plt.close()
    
    return save_path

def plot_price_change_distribution(ticker, price_changes, save_dir):
    """
    绘制股票价格变化率的分布直方图
    
    参数:
        ticker: 股票代码
        price_changes: 价格变化率列表
        save_dir: 图片保存的根目录
    返回:
        str: 保存的图片路径
    """
    plt.figure(figsize=(10, 6))
    plt.hist(price_changes, bins=50, density=True, alpha=0.7)
    plt.axvline(x=0, color='r', linestyle='--', alpha=0.5)
    
    # 添加正态分布拟合曲线
    mu = np.mean(price_changes)
    sigma = np.std(price_changes)
    x = np.linspace(min(price_changes), max(price_changes), 100)
    plt.plot(x, stats.norm.pdf(x, mu, sigma), 'r-', lw=2, 
             label=f'Normal: μ={mu:.2f}, σ={sigma:.2f}')
    
    plt.title(f'{ticker} Price Change Distribution')
    plt.xlabel('Daily Price Change (%)')
    plt.ylabel('Density')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    dist_dir = os.path.join(save_dir, 'pic/distributions')
    os.makedirs(dist_dir, exist_ok=True)
    save_path = os.path.join(dist_dir, f'{ticker}_price_dist.png')
    plt.savefig(save_path)
    plt.close()
    
    return save_path

def plot_volume_analysis(ticker, dates, prices, volumes, save_dir):
    """
    绘制价格和交易量的关系分析图
    
    参数:
        ticker: 股票代码
        dates: 日期列表
        prices: 价格列表
        volumes: 交易量列表
        save_dir: 图片保存的根目录
    返回:
        str: 保存的图片路径
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), height_ratios=[2, 1])
    fig.suptitle(f'{ticker} Price and Volume Analysis')
    
    # 上方子图绘制价格
    ax1.plot(dates, prices, color='blue', linewidth=1.5)
    ax1.set_ylabel('Price')
    ax1.grid(True, alpha=0.3)
    
    # 下方子图绘制成交量
    ax2.bar(dates, volumes, color='green', alpha=0.5)
    ax2.set_ylabel('Volume')
    ax2.grid(True, alpha=0.3)
    
    # 设置x轴标签
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    volume_dir = os.path.join(save_dir, 'pic/volume')
    os.makedirs(volume_dir, exist_ok=True)
    save_path = os.path.join(volume_dir, f'{ticker}_volume.png')
    plt.savefig(save_path)
    plt.close()
    
    return save_path

def plot_correlation_heatmap(tickers, returns_dict, save_dir):
    """
    绘制多支股票收益率的相关性热力图
    
    参数:
        tickers: 股票代码列表
        returns_dict: 包含每支股票收益率的字典
        save_dir: 图片保存的根目录
    返回:
        str: 保存的图片路径
    """
    # 构建收益率矩阵
    returns_df = pd.DataFrame({ticker: returns for ticker, returns in returns_dict.items()})
    
    # 计算相关性矩阵
    corr_matrix = returns_df.corr()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                square=True)
    plt.title('Stock Returns Correlation Matrix')
    
    corr_dir = os.path.join(save_dir, 'pic/correlation')
    os.makedirs(corr_dir, exist_ok=True)
    save_path = os.path.join(corr_dir, 'correlation_heatmap.png')
    plt.savefig(save_path)
    plt.close()
    
    return save_path

def plot_risk_return_scatter(tickers, returns_dict, save_dir):
    """
    绘制风险收益散点图
    
    参数:
        tickers: 股票代码列表
        returns_dict: 包含每支股票收益率的字典
        save_dir: 图片保存的根目录
    返回:
        str: 保存的图片路径
    """
    # 计算每支股票的年化收益率和波动率
    annual_returns = []
    annual_volatilities = []
    
    for ticker in tickers:
        returns = returns_dict[ticker]
        annual_return = np.mean(returns) * 252  # 252个交易日
        annual_volatility = np.std(returns) * np.sqrt(252)
        annual_returns.append(annual_return)
        annual_volatilities.append(annual_volatility)
    
    plt.figure(figsize=(10, 6))
    plt.scatter(annual_volatilities, annual_returns, alpha=0.5)
    
    # 添加标签
    for i, ticker in enumerate(tickers):
        plt.annotate(ticker, (annual_volatilities[i], annual_returns[i]))
    
    plt.title('Risk-Return Analysis')
    plt.xlabel('Annual Volatility')
    plt.ylabel('Annual Return')
    plt.grid(True, alpha=0.3)
    
    risk_dir = os.path.join(save_dir, 'pic/risk')
    os.makedirs(risk_dir, exist_ok=True)
    save_path = os.path.join(risk_dir, 'risk_return_scatter.png')
    plt.savefig(save_path)
    plt.close()
    
    return save_path
