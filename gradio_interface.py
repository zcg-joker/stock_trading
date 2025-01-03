import gradio as gr
import pandas as pd
import torch
import os
from PIL import Image
import warnings
from stock_prediction_lstm import predict, format_feature
from stock_trading.temp import process_stock

warnings.filterwarnings("ignore")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SAVE_DIR = 'tmp/gradio'
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs('tmp/gradio/pic', exist_ok=True)

def get_transaction_csv(transactions_df, save_dir, filename):
    """创建并返回交易记录CSV文件的路径"""
    if transactions_df is not None:
        # 确保文件名有效
        ticker = os.path.splitext(os.path.basename(filename))[0]
        csv_path = os.path.join(save_dir, "transactions", f"{ticker}_transactions.csv")
        return csv_path
    return None

def process_and_predict(file_obj, epochs, batch_size, learning_rate, window_size, 
                       initial_money, agent_iterations, save_dir):
    if file_obj is not None:
        filename = os.path.basename(file_obj.name)
        ticker = os.path.splitext(filename)[0]
        
        # 读取并预测股票数据
        stock_data = pd.read_csv(file_obj.name)
        stock_features = format_feature(stock_data)
        metrics = predict(
            save_dir=save_dir,
            ticker_name=ticker,
            stock_data=stock_data,
            stock_features=stock_features,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate
        )
        
        # 执行交易策略，传入Agent的参数
        trading_results = process_stock(
            ticker, 
            save_dir,
            window_size=window_size,
            initial_money=initial_money,
            iterations=agent_iterations
        )
        
        # 读取图片和交易记录
        prediction_plot = Image.open(f"{save_dir}/pic/predictions/{ticker}_prediction.png")
        loss_plot = Image.open(f"{save_dir}/pic/loss/{ticker}_loss.png")
        earnings_plot = Image.open(f"{save_dir}/pic/earnings/{ticker}_cumulative.png")
        trades_plot = Image.open(f"{save_dir}/pic/trades/{ticker}_trades.png")
        transactions_df = pd.read_csv(f"{save_dir}/transactions/{ticker}_transactions.csv")
        
        return [
            [prediction_plot, loss_plot, earnings_plot, trades_plot],
            metrics['accuracy'] * 100,
            metrics['rmse'],
            metrics['mae'],
            trading_results['total_gains'],
            trading_results['investment_return'],
            trading_results['trades_buy'],
            trading_results['trades_sell'],
            transactions_df
        ]
    return None, 0, 0, 0, 0, 0, 0, 0, None

with gr.Blocks() as demo:
    gr.Markdown("""
    # 智能股票预测与交易Agent
    上传股票价格CSV文件并配置训练参数来运行预测和交易模拟。
    CSV文件必须包含以下列：Date, Open, High, Low, Close, Volume
    """)
    
    save_dir_state = gr.State(value=SAVE_DIR)
    
    with gr.Row():
        file_input = gr.File(label="上传股票CSV文件")
    
    with gr.Tabs():
        with gr.TabItem("LSTM预测参数"):
            with gr.Column():
                lstm_epochs = gr.Slider(minimum=100, maximum=1000, value=500, step=10, 
                                      label="LSTM训练轮数")
                lstm_batch = gr.Slider(minimum=16, maximum=128, value=32, step=16, 
                                     label="LSTM批次大小")
                learning_rate = gr.Slider(minimum=0.0001, maximum=0.01, value=0.001, step=0.0001, 
                                        label="LSTM训练学习率")
        
        with gr.TabItem("交易代理参数"):
            with gr.Column():
                window_size = gr.Slider(minimum=10, maximum=100, value=30, step=5,
                                      label="时间窗口大小")
                initial_money = gr.Number(value=10000, label="初始投资金额 ($)")
                agent_iterations = gr.Slider(minimum=100, maximum=1000, value=500, step=50,
                                          label="代理训练迭代次数")
    
    with gr.Row():
        train_button = gr.Button("开始训练")
    
    with gr.Row():
        output_gallery = gr.Gallery(
            label="分析结果可视化",
            show_label=True,
            elem_id="gallery",
            columns=4, 
            rows=1,
            height="auto",
            object_fit="contain"
        )
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 预测指标")
            accuracy_output = gr.Number(label="预测准确率 (%)")
            rmse_output = gr.Number(label="RMSE (均方根误差)")
            mae_output = gr.Number(label="MAE (平均绝对误差)")
        
        with gr.Column(scale=1):
            gr.Markdown("### 交易指标")
            gains_output = gr.Number(label="总收益 ($)")
            return_output = gr.Number(label="投资回报率 (%)")
            trades_buy_output = gr.Number(label="买入次数")
            trades_sell_output = gr.Number(label="卖出次数")
    
    # 添加一个隐藏的组件来存储当前处理的文件名
    current_file = gr.State(None)

    with gr.Row():
        gr.Markdown("### 交易记录")
        transactions_df = gr.DataFrame(
            headers=["day", "operate", "price", "investment", "total_balance"],
            label="交易详细记录"
        )
    
    with gr.Row():
        # 修改下载按钮部分
        download_button = gr.File(
            label="下载交易记录",
            visible=True,
            interactive=False
        )
    
    results = train_button.click(
        fn=process_and_predict,
        inputs=[
            file_input,
            lstm_epochs,
            lstm_batch,
            learning_rate,
            window_size,
            initial_money,
            agent_iterations,
            save_dir_state
        ],
        outputs=[
            output_gallery,
            accuracy_output,
            rmse_output,
            mae_output,
            gains_output,
            return_output,
            trades_buy_output,
            trades_sell_output,
            transactions_df
        ]
    )
    
    # 添加文件名更新
    file_input.change(
        lambda x: x.name if x else None,
        inputs=[file_input],
        outputs=[current_file]
    )
    
    # 训练完成后更新下载按钮
    results.then(
        fn=get_transaction_csv,
        inputs=[transactions_df, save_dir_state, current_file],
        outputs=[download_button]
    )

demo.launch()