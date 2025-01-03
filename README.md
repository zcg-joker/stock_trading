# Stock Trading AI

基于LSTM预测和强化学习的股票交易AI系统。该系统结合了深度学习的预测能力和强化学习的决策能力，可以自动进行股票价格预测和交易决策。

## 项目特点

- 使用LSTM进行股票价格走势预测
- 采用深度进化策略(Deep Evolution Strategy)进行交易决策
- 完整的数据处理和特征工程
- 可视化界面展示预测和交易结果
- 支持批量处理多支股票

## 环境要求

- Python 3.12+
- Poetry包管理器
- PyTorch (推荐CUDA支持)
- Gradio (用于创建Web界面)

## 安装

1. 克隆项目:
```bash
git clone https://github.com/MilleXi/stock_trading.git
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
- 基于深度进化策略的交易代理
- 自动学习交易策略
- 交易结果分析
- 结果保存在`results/transactions`目录

### 4. 可视化界面
```bash
python gradio_interface.py
```
- 提供Web界面进行交互
- 可视化预测结果和交易决策
- 支持上传股票数据、参数调整和实时预测

## 项目结构

```
stock_trading/
├── data/                   # 存储股票数据
├── results/                # 存储结果
│   ├── predictions/        # 预测结果
│   ├── transactions/       # 交易记录
│   └── pic/               # 可视化图表
├── process_stock_data.py   # 数据处理模块
├── stock_prediction_lstm.py# LSTM预测模块
├── RLagent.py             # 强化学习交易模块
├── visualization.py        # 可视化工具
├── gradio_interface.py     # Web界面
└── README.md              # 项目文档
```

## 主要功能

1. 数据处理
   - 自动下载股票数据
   - 计算技术指标
   - 数据归一化和预处理

2. 价格预测
   - LSTM模型训练
   - 预测准确率评估
   - 预测结果可视化

3. 交易决策
   - 强化学习策略优化
   - 自动交易信号生成
   - 收益率分析

4. 可视化界面
   - 交互式操作
   - 可自由上传数据
   - 可自由调整参数并训练
   - 预测展示
   - 交易结果分析
   - 提供下载Agent交易记录

## 备注

- 项目使用Poetry进行依赖管理
- 如果遇到网络问题，可能需要配置代理来下载股票数据
- 建议使用GPU进行模型训练以提高性能

## 联系方式

如有问题或建议，欢迎在GitHub上提issue。