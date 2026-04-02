from decision_transformer.Colab.utils import create_feature_set
from decision_transformer.Colab.ChartEnv import ChartEnv
import os
import pandas as pd


def build_env():
    symbols = ['EURUSD', 'GBPUSD','USDJPY','USDCHF']
    df_charts = {}

    # build charts
    for file in os.listdir("./decision_transformer/Colab/datafiles"):
        if file.endswith(".pkl"):
            symbol = file.replace(".pkl", "")
            df_charts[symbol] = pd.read_pickle(os.path.join("./decision_transformer/Colab/datafiles", file))
    

    #build datasets
    env_charts = {}
    env_close_prices = {}

    env_test_charts = {}
    env_close_test_prices = {}

    for symbol in symbols:
        chart = df_charts[symbol]
        feature_df,close_df = create_feature_set(chart)

        train_size = int(len(feature_df) * 0.9)

        train_data, train_close_data = feature_df[:train_size ], close_df[:train_size ]
        test_data, test_close_data = feature_df[train_size:],close_df[train_size:]

        train_data.reset_index(drop=True, inplace=True)
        test_data.reset_index(drop=True, inplace=True)
        train_close_data.reset_index(drop=True, inplace=True)
        test_close_data.reset_index(drop=True, inplace=True)

        env_charts[symbol] = train_data
        env_close_prices[symbol] = train_close_data

        env_test_charts[symbol] = test_data
        env_close_test_prices[symbol] = test_close_data
    
    train_env = ChartEnv(chart_dict = env_charts, close_prices= env_close_prices , symbols = symbols,timesteps = 1, episode_length = 1440, recurrent= False, random_start=True) 
    test_env = ChartEnv(chart_dict = env_test_charts, close_prices= env_close_test_prices , symbols = symbols,timesteps = 1, episode_length = 1440, recurrent= False, random_start=True)

    return train_env, test_env

build_env()