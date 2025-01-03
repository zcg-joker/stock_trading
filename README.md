# Stock Trading AI Based RL

基于LSTM预测和强化学习的股票交易AI系统。该系统结合了深度学习的预测能力和强化学习的决策能力，可以自动进行股票价格预测和交易决策。
本项目是对[Stock Trading AI](https://github.com/MilleXi/stock_trading)的进一步优化，在这里也对原作者表示感谢~

## 优化

- 设计了LSTM、BiLSTM、LSTM_with_attention三种模型，并进行了对比
- 优化了结果可视化
- 实现了多种强化学习算法，并进行了对比。实现了真正的基于强化学习的交易决策。

## 环境要求

- Python 3.12+
- Poetry包管理器
- PyTorch (推荐CUDA支持)

## 安装

1. 克隆项目:
```bash
git clone https://github.com/zcg-joker/stock_trading
cd stock_trading
```

2. 使用Poetry安装依赖:
```bash
poetry install
```

如果需要安装PyTorch的特定CUDA版本，请参考[PyTorch官方安装指南](https://pytorch.org/get-started/locally/)。

## 使用说明

项目包含四个主要模块，按以下顺序运行：

### 1. 数据获取与处理
```bash
python process_stock_data.py
```
- 从Yahoo Finance下载股票数据
- 计算技术指标（如MA, RSI等）
- 数据预处理和清洗，包括去除缺失值、归一化等
- 结果保存在`data`目录中，包含处理后的历史股票数据以及技术指标

### 2. LSTM预测模型
```bash
python stock_prediction_lstm.py
```
- 使用LSTM模型预测股票价格
- 模型训练、验证、评估
- 预测结果可视化
- 结果保存在`results/predictions`目录

### 3. 强化学习交易代理
```bash
python RLagent.py
```
- 自动学习交易策略
- 交易结果分析
- 结果保存在`results/rlresults`目录

## 联系方式

如有问题或建议，欢迎在GitHub上提issue。