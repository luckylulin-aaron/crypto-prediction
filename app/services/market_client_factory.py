"""Explicit construction boundary for external market clients."""

from dataclasses import dataclass


@dataclass(frozen=True)
class CryptoMarketClients:
    coinbase: object
    binance: object


class MarketClientFactory:
    def __init__(self, coinbase_class, binance_class, stock_class):
        self._coinbase_class = coinbase_class
        self._binance_class = binance_class
        self._stock_class = stock_class

    def crypto(self, *, coinbase_key, coinbase_secret, binance_key, binance_secret):
        return CryptoMarketClients(
            coinbase=self._coinbase_class(key=coinbase_key, secret=coinbase_secret),
            binance=self._binance_class(api_key=binance_key, api_secret=binance_secret),
        )

    def stocks(self, tickers):
        return self._stock_class(tickers=list(tickers))
