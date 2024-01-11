import math
from io import TextIOWrapper
from typing import Any, Dict, Optional, Tuple

import gym
import numpy as np
from gym import spaces

from stocktypes.StockTypes import Rewards, StockData, StockPortfolio, TransactionData


class DiscreteSingleStockEnvironment(gym.Env):
    initial_balance = 1000
    balance: float = 1000

    portfolio: StockPortfolio = {}
    transaction: TransactionData = {}
    rewards: Rewards = {}

    log_returns: np.ndarray = []

    def __init__(
        self,
        ticker: str,
        data: StockData,
        log_file: Optional[TextIOWrapper],
        transaction_rate: float = 0.001,
    ):
        super(DiscreteSingleStockEnvironment, self).__init__()

        self.ticker = ticker
        self.data = data
        self.log_file = log_file
        self.transaction_rate = transaction_rate

        # Max steps is econfidenceual to length of data
        self.current_step: int = 1
        self.max_steps: int = len(data["share_price"])

        observation = self.reset()
        shape = observation.shape

        # Observation space, holds historical data
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=shape)
        # Action space, 0 = sell, 1 = hold, 2 = buy
        self.action_space = spaces.Discrete(3)

    def reset(self):
        self.current_step = 1
        self.balance = self.initial_balance

        self.portfolio["num_shares"] = 0
        self.portfolio["num_shares_bought"] = 0
        self.portfolio["num_shares_sold"] = 0
        self.portfolio["value"] = 0.0
        self.portfolio["avg_buy_price"] = 0.0
        self.portfolio["avg_sell_price"] = 0.0

        return self._get_observation()

    def step(self, action: int, confidence: float):
        assert (
            0 <= confidence and confidence <= 1
        ), f"Confidence should be within [0, 1], got {confidence}"

        self._update_log_returns()

        # Get previous day and current day prices
        curr_price = self.data["share_price"][self.current_step]
        last_price = self.data["share_price"][self.current_step - 1]

        # Initialize rewards
        self.rewards["price_change"] = (
            self._get_sharpe_ratio()
            * self._pct_change(last_price, curr_price)
            * (-1 if action == 2 else 1)
        )
        self.rewards["transaction_improvement"] = 0
        self.rewards["buy_sell_diff"] = 0
        self.rewards["portfolio_net"] = 0
        self.rewards["penalty"] = 0

        if action == 0 and self.portfolio["num_shares"] == 0:
            # Invalid sell
            self.transaction["type"] = "NULL_SELL"
            self.transaction["num_shares"] = 0
            self.rewards["penalty"] -= 10
        elif action == 0:
            # Sell
            self.transaction["type"] = "SELL"
            self.transaction["num_shares"] = max(
                1, math.ceil(confidence * self.portfolio["num_shares"])
            )
            sale = self.transaction["num_shares"] * curr_price
            transaction_fee = self.transaction_rate * self.transaction["num_shares"]

            # Update portfolio
            prev_avg_sell_price = self.portfolio["avg_sell_price"]
            prev_num_shares_sold = self.portfolio["num_shares_sold"]
            self.portfolio["num_shares_sold"] += self.transaction["num_shares"]
            self.portfolio["num_shares"] -= self.transaction["num_shares"]
            self.balance += sale - transaction_fee

            # Update average sell price
            self.portfolio["avg_sell_price"] = (
                prev_num_shares_sold * prev_avg_sell_price + sale
            ) / self.portfolio["num_shares_sold"]

            # Update transaction improvement reward
            if prev_num_shares_sold != 0:
                self.rewards["transaction_improvement"] = self._pct_change(
                    prev_avg_sell_price, self.portfolio["avg_sell_price"]
                )
        elif action == 2 and self.balance / curr_price < 1:
            self.transaction["type"] = "NULL_BUY"
            self.transaction["num_shares"] = 0
            self.rewards["penalty"] -= 10
        elif action == 2:
            # Buy
            self.transaction["type"] = "BUY"
            self.transaction["num_shares"] = max(
                1, math.floor(confidence * self.balance / curr_price)
            )
            cost = self.transaction["num_shares"] * curr_price
            transaction_fee = self.transaction_rate * self.transaction["num_shares"]

            # Update portfolio
            prev_num_shares_bought = self.portfolio["num_shares_bought"]
            prev_avg_buy_price = self.portfolio["avg_buy_price"]
            self.portfolio["num_shares_bought"] += self.transaction["num_shares"]
            self.portfolio["num_shares"] += self.transaction["num_shares"]
            self.balance -= cost + transaction_fee

            # Update average buy price
            self.portfolio["avg_buy_price"] = (
                prev_num_shares_bought * prev_avg_buy_price + cost
            ) / self.portfolio["num_shares_bought"]

            # Update transaction improvement reward
            if prev_num_shares_bought != 0:
                self.rewards["transaction_improvement"] = self._pct_change(
                    self.portfolio["avg_buy_price"], prev_avg_buy_price
                )
        else:
            # Hold
            self.transaction["type"] = "HOLD"
            self.transaction["num_shares"] = self.portfolio["num_shares"]
            self.rewards["penalty"] -= 3

        # Update buy/sell difference reward
        if self.transaction["type"] != "HOLD" and self.portfolio["num_shares_sold"] > 0:
            self.rewards["buy_sell_diff"] = (
                self._pct_change(
                    self.portfolio["avg_buy_price"], self.portfolio["avg_sell_price"]
                )
                + self._pct_change(self.portfolio["avg_buy_price"], curr_price)
                + self._pct_change(curr_price, self.portfolio["avg_sell_price"])
            )

        # Update portfolio net reward
        self.portfolio["value"] = self.portfolio["num_shares"] * curr_price
        self.rewards["portfolio_net"] = self._pct_change(
            self.initial_balance, self.portfolio["value"] + self.balance
        )

        # Calculate total reward!
        reward = self._calculate_reward()

        # Log transaction information
        if self.current_step % 5 == 0:
            self._log(curr_price, reward)

        # Increment step
        self.current_step += 1
        done = self.current_step >= self.max_steps - 1

        return self._get_observation(), reward, done, {}

    def _calculate_reward(self) -> float:
        # Reward multipliers
        if self.transaction["type"] == "SELL":
            if self.rewards["price_change"] > 0:
                self.rewards["price_change"] *= 2
            if self.rewards["buy_sell_diff"] > 0:
                self.rewards["buy_sell_diff"] *= 2
        elif self.transaction["type"] == "BUY":
            pass
        elif self.transaction["type"] == "HOLD":
            if self.rewards["price_change"] < 0:
                self.rewards["price_change"] *= 2

        return sum(self.rewards.values())

    def _get_observation(self):
        # Next states
        states = [
            # Portfolio state
            self.portfolio["num_shares"],
            self.balance,
            # Current price
            self.data["share_price"][self.current_step],
            # Technical indicators
            self.data["volume"][self.current_step],
            self.data["capitalization"][self.current_step],
            self.data["dividend_yield"][self.current_step],
            self.data["price_to_book"][self.current_step],
            self.data["price_to_earnings"][self.current_step],
        ]

        return np.array(states)

    def _log(self, curr_price: float, reward: float):
        if self.log_file is None:
            return
        self.log_file.write(f"Current step: {self.current_step}\n")

        self.log_file.write("TRANSACTION:\n")
        self.log_file.write(f"• Type: {self.transaction['type']}\n")
        self.log_file.write(f"• Volume: {self.transaction['num_shares']}\n")
        self.log_file.write(f"• Price: ${curr_price}\n")

        self.log_file.write("PORTFOLIO:\n")
        self.log_file.write(f"• Gross: ${self.portfolio['value'] + self.balance}\n")
        self.log_file.write(f"• Invested: ${self.portfolio['value']}\n")
        self.log_file.write(f"• Shares owned: {self.portfolio['num_shares']}\n")
        self.log_file.write(f"• Avg. buy: ${self.portfolio['avg_buy_price']}\n")
        self.log_file.write(f"• Avg. sell: ${self.portfolio['avg_sell_price']}\n")

        self.log_file.write(f"REWARDS:\n")
        self.log_file.write(f"• Total: {reward}\n")
        self.log_file.write(
            f"• Risk-adjusted price change: {self.rewards['price_change']}\n"
        )
        self.log_file.write(
            f"• Transaction improvement: {self.rewards['transaction_improvement']}\n"
        )
        self.log_file.write(f"• Buy/Sell difference: {self.rewards['buy_sell_diff']}\n")
        self.log_file.write(f"• Portfolio gain/loss: {self.rewards['portfolio_net']}\n")
        self.log_file.write(f"• Penalty: {self.rewards['penalty']}\n\n")

    def _update_log_returns(self):
        price_history = np.array(
            self.data["share_price"][max(0, self.current_step - 21) : self.current_step]
        )
        self.log_returns = np.log(price_history[1:] / price_history[:-1])

    def _get_sharpe_ratio(self):
        std_returns = self._get_volatility()

        if std_returns != 0 and len(self.log_returns) >= 2:
            return np.mean(self.log_returns) / std_returns
        else:
            return 0

    def _get_volatility(self):
        return np.std(self.log_returns)

    def _pct_change(self, old_value, new_value):
        return 0 if old_value == 0 else 100 * (new_value - old_value) / old_value
