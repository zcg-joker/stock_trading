import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
import pickle
import warnings
from visualization import (
    plot_stock_prediction,
    plot_training_loss,
    plot_cumulative_earnings,
    plot_accuracy_comparison,
    plot_price_change_distribution,
    plot_volume_analysis,
    plot_correlation_heatmap,
    plot_risk_return_scatter
)
import json
import logging

warnings.filterwarnings("ignore", category=FutureWarning)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

class BiLSTMModel(nn.Module):
    """
    双向LSTM模型
    """
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2):
        super(BiLSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_size, 
            hidden_size, 
            num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout
        )
        # 由于是双向LSTM,hidden_size需要乘2
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
        self.fc = nn.Linear(hidden_size * 2, output_size)
        
    def attention_net(self, lstm_output):
        # lstm_output shape: (batch_size, seq_len, hidden_size * 2)
        attn_weights = self.attention(lstm_output)
        # soft_attn_weights shape: (batch_size, seq_len, 1)
        soft_attn_weights = torch.softmax(attn_weights, dim=1)
        # context shape: (batch_size, hidden_size * 2)
        context = torch.sum(lstm_output * soft_attn_weights, dim=1)
        return context
        
    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_size).to(device)
        
        # lstm_output shape: (batch_size, seq_len, hidden_size * 2)
        lstm_output, _ = self.lstm(x, (h0, c0))
        
        # 使用注意力机制
        attn_output = self.attention_net(lstm_output)
        
        # 最终输出
        out = self.fc(attn_output)
        return out

class LSTMWithAttention(nn.Module):
    """
    带注意力机制的LSTM模型
    """
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2):
        super(LSTMWithAttention, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=dropout)
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
        self.fc = nn.Linear(hidden_size, output_size)
        
    def attention_net(self, lstm_output):
        attn_weights = self.attention(lstm_output)
        soft_attn_weights = torch.softmax(attn_weights, dim=1)
        context = torch.sum(lstm_output * soft_attn_weights, dim=1)
        return context
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        
        lstm_output, _ = self.lstm(x, (h0, c0))
        attn_output = self.attention_net(lstm_output)
        out = self.fc(attn_output)
        return out

def get_stock_data(ticker, data_dir='data'):
    file_path = os.path.join(data_dir, f'{ticker}.csv')
    data = pd.read_csv(file_path, index_col='Date', parse_dates=True)
    return data

def format_feature(data):
    features = [
        'Volume', 'Year', 'Month', 'Day', 'MA5', 'MA10', 'MA20', 'RSI', 'MACD',
        'VWAP', 'SMA', 'Std_dev', 'Upper_band', 'Lower_band', 'ATR',
        'Close_yes', 'Open_yes', 'High_yes', 'Low_yes'
    ]
    
    # 计算 Relative_Performance
    data['Relative_Performance'] = data['Close'].pct_change().fillna(0)  # 计算相对表现率，填充缺失值为0
    
    # 确保特征中包含 Relative_Performance
    features.append('Relative_Performance')
    
    X = data[features].iloc[1:]  # 这里的 iloc[1:] 是为了避免 NaN
    y = data['Close'].pct_change().iloc[1:]
    return X, y

def prepare_data(data, n_steps):
    X, y = [], []
    for i in range(len(data) - n_steps):
        X.append(data[i:i + n_steps])
        y.append(data[i + n_steps])
    return np.array(X), np.array(y)

def visualize_predictions(ticker, data, predict_result, test_indices, predictions, actual_percentages, save_dir):
    actual_prices = data['Close'].loc[test_indices].values
    predicted_prices = np.array(predictions)
    
    mse = np.mean((predicted_prices - actual_prices) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(predicted_prices - actual_prices))
    accuracy = 1 - np.mean(np.abs(predicted_prices - actual_prices) / actual_prices)
    
    metrics = {'rmse': rmse, 'mae': mae, 'accuracy': accuracy}
    plot_stock_prediction(ticker, test_indices, actual_prices, predicted_prices, metrics, save_dir)
    
    return metrics

def train_and_predict_lstm(ticker, data, X, y, save_dir, model_type='lstm', n_steps=60, 
                          num_epochs=500, batch_size=32, learning_rate=0.001,
                          hidden_size=50, num_layers=2, dropout=0.2):
    # 数据归一化和准备部分
    scaler_y = MinMaxScaler()
    scaler_X = MinMaxScaler()
    scaler_y.fit(y.values.reshape(-1, 1))
    y_scaled = scaler_y.transform(y.values.reshape(-1, 1))
    X_scaled = scaler_X.fit_transform(X)

    X_train, y_train = prepare_data(X_scaled, n_steps)
    y_train = y_scaled[n_steps-1:-1]

    train_per = 0.8
    split_index = int(train_per * len(X_train))
    X_val = X_train[split_index-n_steps+1:]
    y_val = y_train[split_index-n_steps+1:]
    X_train = X_train[:split_index]
    y_train = y_train[:split_index]

    # PyTorch数据准备
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32).to(device)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = create_model(
        model_type=model_type,
        input_size=X_train.shape[2],
        hidden_size=hidden_size,
        num_layers=num_layers,
        output_size=1,
        dropout=dropout
    ).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)

    train_losses = []
    val_losses = []

    with tqdm(total=num_epochs, desc=f"Training {ticker}", unit="epoch") as pbar:
        for epoch in range(num_epochs):
            # 训练和验证循环
            model.train()
            epoch_train_loss = 0
            for inputs, targets in train_loader:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_train_loss += loss.item()

            avg_train_loss = epoch_train_loss / len(train_loader)
            train_losses.append(avg_train_loss)

            model.eval()
            epoch_val_loss = 0
            with torch.no_grad():
                for inputs, targets in val_loader:
                    outputs = model(inputs)
                    val_loss = criterion(outputs, targets)
                    epoch_val_loss += val_loss.item()

            avg_val_loss = epoch_val_loss / len(val_loader)
            val_losses.append(avg_val_loss)

            pbar.set_postfix({"Train Loss": avg_train_loss, "Val Loss": avg_val_loss})
            pbar.update(1)
            scheduler.step()

    # 使用可视化工具绘制损失曲线
    plot_training_loss(ticker, train_losses, val_losses, save_dir)

    # 预测
    model.eval()
    predictions = []
    test_indices = []
    predict_percentages = []
    actual_percentages = []

    with torch.no_grad():
        for i in range(1 + split_index, len(X_scaled) + 1):
            x_input = torch.tensor(X_scaled[i - n_steps:i].reshape(1, n_steps, X_train.shape[2]), 
                                 dtype=torch.float32).to(device)
            y_pred = model(x_input)
            y_pred = scaler_y.inverse_transform(y_pred.cpu().numpy().reshape(-1, 1))
            predictions.append((1 + y_pred[0][0]) * data['Close'].iloc[i - 2])
            test_indices.append(data.index[i - 1])
            predict_percentages.append(y_pred[0][0] * 100)
            actual_percentages.append(y[i - 1] * 100)

    # 使用可视化工具绘制累积收益率曲线
    plot_cumulative_earnings(ticker, test_indices, actual_percentages, predict_percentages, save_dir)

    predict_result = {str(date): pred / 100 for date, pred in zip(test_indices, predict_percentages)}
    return predict_result, test_indices, predictions, actual_percentages

def save_predictions_with_indices(ticker, test_indices, predictions, save_dir):
    df = pd.DataFrame({
        'Date': test_indices,
        'Prediction': predictions
    })

    file_path = os.path.join(save_dir, 'predictions', f'{ticker}_predictions.pkl')
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'wb') as file:
        pickle.dump(df, file)

    print(f'Saved predictions for {ticker} to {file_path}')

def predict(ticker_name, stock_data, stock_features, save_dir, model_type='lstm', epochs=500, batch_size=32, learning_rate=0.001):
    """
    执行股票预测并生成可视化分析
    
    参数:
        ticker_name: 股票代码
        stock_data: 股票数据
        stock_features: 特征数据
        save_dir: 保存目录
        epochs: 训练轮数
        batch_size: 批次大小
        learning_rate: 学习率
    """
    logger = logging.getLogger()
    logger.info(f"\n开始处理股票: {ticker_name}")
    
    try:
        data = stock_data
        X, y = stock_features
        
        # 训练模型并获取预测结果
        predict_result, test_indices, predictions, actual_percentages = train_and_predict_lstm(
            ticker_name, data, X, y, save_dir, model_type=model_type, 
            num_epochs=epochs, batch_size=batch_size, learning_rate=learning_rate
        )
        
        # 生成预测可视化和计算指标
        metrics = visualize_predictions(ticker_name, data, predict_result, test_indices, 
                                     predictions, actual_percentages, save_dir)
        
        # 保存预测结果
        save_predictions_with_indices(ticker_name, test_indices, predictions, save_dir)
        
        # 保存当前股票的预测指标
        metrics_dir = os.path.join(save_dir, 'metrics')
        os.makedirs(metrics_dir, exist_ok=True)
        metrics_file = os.path.join(metrics_dir, f'{ticker_name}_metrics.json')
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f)
        
        logger.info(f"股票 {ticker_name} 预测完成")
        logger.info(f"预测指标: {metrics}")
        
        return metrics
        
    except Exception as e:
        logger.error(f"处理股票 {ticker_name} 时发生错误: {str(e)}")
        return None

def create_model(model_type, input_size, hidden_size, num_layers, output_size, dropout=0.2):
    """
    创建指定类型的模型
    
    参数:
        model_type: 模型类型 ('lstm', 'bilstm', 'lstm_attention')
        input_size: 输入特征维度
        hidden_size: 隐藏层大小
        num_layers: LSTM层数
        output_size: 输出维度
        dropout: dropout率
    """
    if model_type == 'lstm':
        return LSTMModel(input_size, hidden_size, num_layers, output_size, dropout)
    elif model_type == 'bilstm':
        return BiLSTMModel(input_size, hidden_size, num_layers, output_size, dropout)
    elif model_type == 'lstm_attention':
        return LSTMWithAttention(input_size, hidden_size, num_layers, output_size, dropout)
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")

def setup_logger(save_dir):
    """
    设置日志记录器
    """
    # 创建日志目录
    log_dir = os.path.join(save_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    # 配置日志记录器
    log_file = os.path.join(log_dir, f'training_{pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")}.log')
    
    # 创建处理程序
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    console_handler = logging.StreamHandler()
    
    # 设置格式
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # 配置根日志记录器
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

if __name__ == "__main__":
    # 实验参数配置
    config = {
        # 数据参数
        'data_dir': 'data',  # 存放股票数据的目录
        'save_dir': 'results',  # 存放结果和输出的目录
        
        # 模型参数
        'model_configs': {
            'default': 'lstm_attention',  # 默认模型类型，指定使用的模型（如LSTM、双向LSTM或带注意力机制的LSTM）
            'hidden_size': 50,  # LSTM隐藏层的大小
            'num_layers': 2,  # LSTM的层数
            'dropout': 0.2  # dropout率，用于防止过拟合
        },
        
        # 训练参数
        'train_params': {
            'n_steps': 60,           # 时间步长，输入序列的长度
            'epochs': 50,           # 训练的轮数
            'batch_size': 64,        # 每个批次的样本数量
            'learning_rate': 0.001,  # 学习率，控制模型更新的步长
            'train_split': 0.7       # 训练集与验证集的比例，0.8表示80%的数据用于训练，20%用于验证
        },
        
        # 股票列表
        'tickers': [
            'AAPL', 'MSFT', 'GOOGL', 
            'AMZN', 'TSLA',  # 科技股
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
        ],
        
        # 特定股票的模型配置
        'stock_specific_configs': {
            # 'AAPL': {
            #     'model_type': 'bilstm',
            #     'hidden_size': 64,
            #     'num_layers': 3
            # },
            # 'GOOGL': {
            #     'model_type': 'lstm_attention',
            #     'hidden_size': 128,
            #     'num_layers': 4
            # }
        }
    }
    
    # 设置日志记录器
    logger = setup_logger(config['save_dir'])
    logger.info("开始股票预测实验")
    logger.info(f"实验配置:\n{json.dumps(config, indent=4, ensure_ascii=False)}")
    
    # 用于存储所有股票的预测指标
    all_metrics = {}
    returns_dict = {}
    
    for ticker_name in config['tickers']:
        try:
            # 获取股票特定配置
            stock_config = config['stock_specific_configs'].get(ticker_name, {})
            
            # 获取数据
            stock_data = get_stock_data(ticker_name, config['data_dir'])
            stock_features = format_feature(stock_data)
            
            # 计算收益率并存储
            returns = stock_data['Close'].pct_change().dropna()
            returns_dict[ticker_name] = returns
            
            # 执行预测
            metrics = predict(
                ticker_name=ticker_name,
                stock_data=stock_data,
                stock_features=stock_features,
                save_dir=config['save_dir'],
                model_type=stock_config.get('model_type', config['model_configs']['default']),
                epochs=config['train_params']['epochs'],
                batch_size=config['train_params']['batch_size'],
                learning_rate=config['train_params']['learning_rate']
            )
            
            if metrics is not None:
                all_metrics[ticker_name] = metrics
            
        except Exception as e:
            logger.error(f"处理股票 {ticker_name} 时发生错误: {str(e)}")
            continue
    
    # 生成准确率对比图
    if all_metrics:
        plot_accuracy_comparison(all_metrics, config['save_dir'])
        logger.info("已生成准确率对比图")
    
    try:
        # 生成相关性热力图
        plot_correlation_heatmap(config['tickers'], returns_dict, config['save_dir'])
        
        # 生成风险收益散点图
        plot_risk_return_scatter(config['tickers'], returns_dict, config['save_dir'])
        
        logger.info(f"\n分析完成! 所有结果已保存到 {config['save_dir']}")
        
    except Exception as e:
        logger.error(f"生成汇总分析图表时发生错误: {str(e)}")