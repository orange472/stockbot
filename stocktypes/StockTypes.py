from collections import namedtuple
from datetime import datetime
from typing import List, Literal, Optional, TypedDict, Union


class StockData(TypedDict):
    date: List[datetime]
    share_price: List[float]
    normalized_price: List[float]
    moving_average: List[float]
    volume: List[float]
    capitalization: List[float]
    dividend_yield: List[float]
    price_to_book: List[float]
    price_to_earnings: List[float]


class StockPortfolio(TypedDict):
    num_shares: int
    num_shares_bought: float
    num_shares_sold: float
    value: float
    avg_buy_price: float
    avg_sell_price: float
    industry: Optional[str]


class Rewards(TypedDict):
    price_change: float
    transaction_improvement: float
    buy_sell_diff: float
    portfolio_net: float
    penalty: float


class TransactionData(TypedDict):
    type: Literal["BUY", "SELL", "HOLD", "NULL_SELL", "NULL_BUY"]
    num_shares: Union[float, int]


class DRQNAgentProps(TypedDict):
    load_model: Optional[str]
    load_target: Optional[str]
    save_model: Optional[str]
    save_target: Optional[str]


class DRQNModelProps(TypedDict):
    model_type: Literal["model", "target"]
    load_path: Optional[str]
    save_path: Optional[str]


Experience = namedtuple(
    "Experience",
    [
        "state",
        "action",
        "reward",
        "next_state",
        "done",
    ],
)

