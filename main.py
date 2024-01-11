import json
import os
from typing import List, Literal, Optional

import numpy as np
import pandas as pd
import requests

from environments import DiscreteSingleStockEnvironment
from models import DRQN_agent
from stocktypes.StockTypes import DRQNAgentProps, StockData


def main(mode: Literal["continuous", "discrete"]):
    if mode == "Discrete":
        print("Mode: discrete")
    if mode == "continuous":
        raise Exception("Continuous action space not implemented yet!")

    tickers: List[str] = []
    stocks_data: List[StockData] = []
    columns = [
        "date",
        "share_price",
        "volume",
        "capitalization",
        "dividend_yield",
        "price_to_book",
        "price_to_earnings",
    ]

    dir_path = "./test-market"
    max_stocks = 8

    # Get historical stock data
    for i, file_name in enumerate(os.listdir(dir_path)):
        if not file_name.endswith((".csv", ".xlsx", ".xls")):
            continue
        file_path = os.path.join(dir_path, file_name)

        # Get ticker (ticker is in the form MARKET:TICKER), which sould start at row 3
        ticker_df: pd.DataFrame = pd.read_excel(file_path, skiprows=2, nrows=1)
        tickers.append(ticker_df.at[0, ticker_df.columns[0]])

        # Get historical data, which should start at row 54
        table: pd.DataFrame = pd.read_excel(file_path, skiprows=53)

        # Add historical data to stocks_data
        formatted_data: StockData = {}
        for j, column_name in enumerate(table.columns):
            data = table[column_name].tolist()[::-1]
            if j == 0:
                formatted_data["date"] = data
            elif j == 1:
                formatted_data[columns[j]] = data
                formatted_data["normalized_price"] = normalize_data(data)
                formatted_data["moving_average"] = get_moving_average(data)
            else:
                formatted_data[columns[j]] = data

        stocks_data.append(formatted_data)

        if i >= max_stocks - 1:
            break

    agent_props: DRQNAgentProps = {}
    agent_props["load_model"] = "saved_models/model.h5"
    agent_props["save_model"] = "saved_models"
    agent_props["load_target"] = "saved_models/target.h5"
    agent_props["save_target"] = "saved_models"
    agent: Optional[DRQN_agent] = None

    results = open("logs/training_results.txt", "w").close()
    results = open("logs/training_results.txt", "a")

    for i in range(len(tickers)):
        results.write(
            "({}/{}) Training model on {}\n".format(i + 1, len(tickers), tickers[i])
        )
        results.flush()

        # Open log file
        log_file = open(f"logs/{tickers[i]}.txt", "w").close()
        log_file = open(f"logs/{tickers[i]}.txt", "a")

        # Create environment with ticker
        env = DiscreteSingleStockEnvironment(
            ticker=tickers[i], data=stocks_data[i], log_file=log_file
        )

        # Create agent
        if agent is None:
            agent = DRQN_agent(env, agent_props)
        else:
            agent.env = env
        agent.train(max_episodes=50)

        results.write("Final episode:\n")
        results.write("Final value: {}\n".format(env.balance + env.portfolio["value"]))
        results.write("Balance: {}\n".format(env.balance))
        results.write("Portfolio: {}\n\n".format(json.dumps(env.portfolio)))
        results.flush()

        # Close log files
        log_file.close()

    results.close()


def normalize_data(data: list):
    try:
        min_val = np.min(data)
        max_val = np.max(data)
        return (data - min_val) / (max_val - min_val)
    except:
        print("Could not normalize data!")
        return data


def get_moving_average(data: list):
    return np.convolve(np.array(data), np.ones(21) / 21, mode="valid")


def get_sentiment(self, time_from: str):
    url = f'https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={self.ticker}&apikey={os.environ["ALPHA_VANTAGE_API_KEY"]}&time_from={time_from}&limit=50'
    res = requests.get(url)
    return res.json()


if __name__ == "__main__":
    main("discrete")
