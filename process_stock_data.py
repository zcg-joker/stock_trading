import os
import pandas as pd
import numpy as np
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

def calculate_technical_indicators(data, start_date=None, end_date=None):
    """
    计算股票的技术指标
    
    参数:
        data: DataFrame, 包含OHLCV数据的DataFrame
        start_date: str, 开始日期 (可选，用于相对表现计算)
        end_date: str, 结束日期 (可选，用于相对表现计算)
    
    返回:
        DataFrame: 添加了技术指标的数据
    """
    # 添加日期特征
    data['Year'] = data.index.year
    data['Month'] = data.index.month
    data['Day'] = data.index.day
    
    # 移动平均线
    data['MA5'] = data['Close'].shift(1).rolling(window=5).mean()
    data['MA10'] = data['Close'].shift(1).rolling(window=10).mean()
    data['MA20'] = data['Close'].shift(1).rolling(window=20).mean()
    
    # RSI指标
    delta = data['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    data['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD指标
    exp1 = data['Close'].ewm(span=12, adjust=False).mean()
    exp2 = data['Close'].ewm(span=26, adjust=False).mean()
    data['MACD'] = exp1 - exp2
    data['Signal_Line'] = data['MACD'].ewm(span=9, adjust=False).mean()
    data['MACD_Histogram'] = data['MACD'] - data['Signal_Line']
    
    # VWAP指标
    data['VWAP'] = (data['Close'] * data['Volume']).cumsum() / data['Volume'].cumsum()
    
    # 布林带
    period = 20
    data['SMA'] = data['Close'].rolling(window=period).mean()
    data['Std_dev'] = data['Close'].rolling(window=period).std()
    data['Upper_band'] = data['SMA'] + 2 * data['Std_dev']
    data['Lower_band'] = data['SMA'] - 2 * data['Std_dev']
    
    # 相对大盘表现
    if start_date and end_date:
        benchmark_data = yf.download('SPY', start=start_date, end=end_date)['Close']
        data['Relative_Performance'] = (data['Close'] / benchmark_data.values) * 100
    
    # ROC指标
    data['ROC'] = data['Close'].pct_change(periods=1) * 100
    
    # ATR指标
    high_low_range = data['High'] - data['Low']
    high_close_range = abs(data['High'] - data['Close'].shift(1))
    low_close_range = abs(data['Low'] - data['Close'].shift(1))
    true_range = pd.concat([high_low_range, high_close_range, low_close_range], axis=1).max(axis=1)
    data['ATR'] = true_range.rolling(window=14).mean()
    
    # 前一天数据
    data[['Close_yes', 'Open_yes', 'High_yes', 'Low_yes']] = data[['Close', 'Open', 'High', 'Low']].shift(1)
    
    # 删除缺失值
    data = data.dropna()
    
    return data

def get_stock_data(ticker, start_date, end_date):
    """
    获取并处理单个股票的数据
    
    参数:
        ticker: 股票代码
        start_date: 起始日期
        end_date: 结束日期
    返回:
        处理后的股票数据DataFrame
    """
    # 下载股票数据
    data = yf.download(ticker, start=start_date, end=end_date)
    print(data)
    
    # 计算技术指标
    data = calculate_technical_indicators(data, start_date, end_date)
    
    return data

def clean_csv_files(folder_path):
    """
    清理CSV文件，删除多余行并重命名列
    
    参数:
        folder_path: CSV文件所在文件夹路径
    """
    for filename in os.listdir(folder_path):
        if filename.endswith('.csv'):
            file_path = os.path.join(folder_path, filename)
            df = pd.read_csv(file_path)
            
            # 删除第二行和第三行
            df = df.drop([0, 1]).reset_index(drop=True)
            
            # 重命名列
            df = df.rename(columns={'Price': 'Date'})
            
            # 保存修改后的文件
            df.to_csv(file_path, index=False)
    print("所有文件处理完成！")

def main():
    """主函数：执行数据收集和处理流程"""
    # 股票分类列表
    tickers = [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA',  # 科技股
            'JPM', 'BAC', 'GS', 'C', 'WFC',            # 金融股
            'JNJ', 'PFE', 'MRK', 'ABBV', 'BMY',        # 医药股  
            'XOM', 'CVX', 'COP', 'SLB', 'BKR',         # 能源股
            'DIS', 'NFLX', 'CMCSA', 'NKE', 'SBUX',    # 消费股
            'CAT', 'DE', 'MMM', 'GE', 'HON',          # 工业股
            'NFLX', 'NVDA', 'AMD', 'INTU', 'CRM',      # 更多科技股
            'V', 'Z', 'TSN', 'KO', 'PEP',              # 更多消费股
            'T', 'MO', 'VZ', 'QCOM',                    # 更多电信股
            'BA', 'LMT', 'NOC', 'RTX',                  # 更多国防股
            'UNH', 'AET', 'CI', 'HUM',                  # 更多医疗股
            'MPC', 'VLO', 'PSX',                        # 更多能源股
            'NKE', 'LULU', 'TGT',                      # 更多零售股
            'WMT', 'COST', 'HD', 'LOW'                 # 更多零售股
            ]   

    # 设置参数
    START_DATE = '2020-01-01'
    END_DATE = '2024-01-01'
    NUM_FEATURES_TO_KEEP = 9
    
    # 创建数据文件夹
    data_folder = 'data'
    os.makedirs(data_folder, exist_ok=True)
    
    # 获取并保存所有股票数据
    print("开始下载和处理股票数据...")
    for ticker in tickers:
        try:
            print(f"处理 {ticker} 中...")
            stock_data = get_stock_data(ticker, START_DATE, END_DATE)
            stock_data.to_csv(f'{data_folder}/{ticker}.csv')
            print(f"{ticker} 处理完成")
        except Exception as e:
            print(f"处理 {ticker} 时出错: {str(e)}")
        break
        
    
    # 清理CSV文件
    print("\n开始清理CSV文件...")
    clean_csv_files(data_folder)
    print("数据处理完成！")

if __name__ == "__main__":
    main()