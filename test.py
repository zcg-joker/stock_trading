import pandas as pd


df = pd.read_pickle(f'results/predictions/AAPL_predictions.pkl')
# print(f"\nProcessing {ticker}")
close = df.Prediction.values.tolist()

df.to_csv('AAPL_predictions.csv')
