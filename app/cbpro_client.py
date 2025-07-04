# built-in packages
import datetime
from typing import Any, List

# third-party packages
from coinbase.rest import RESTClient
# customized packages
from config import *
from util import *
from logger import get_logger


class CBProClient:

    def __init__(self, key: str='', secret: str=''):
        self.logger = get_logger(__name__)
        # Create the RESTClient from coinbase-advanced-py
        if len(key) > 0 and len(secret) > 0:
            self.rest_client = RESTClient(api_key=key, api_secret=secret)
        else:
            self.rest_client = RESTClient()  # Will use env vars if set

    @timer
    def get_cur_rate(self, name: str):
        '''For a given currency name, return its latest hour\'s price.

        :argument
            name (str): Currency name.

        :return
            res (float): Value in USD.

        :raise
            ConnectionError (Exception):
        '''
        try:
            res = self.rest_client.get_product_24hr_stats(name)['last']
        except Exception as e:
            raise ConnectionError(f'cannot get current rate for {name}')

        res = float(res)
        return res

    @timer
    def get_historic_data(self, name: str, grans: int):
        '''Get historic rates.

        :argument
            name (str): Currency name.
            grans (int): Granularity, i.e. desired time slice in seconds; can only be one of the followings:
                 {60 (1 min), 300 (5 mins), 900 (15 mins), 3600 (1hrs), 21600 (6hrs), 86400 (24hrs)}

        :return
            res (List[List[float, str, float, float, float]]): Data stream of various prices and dates.

        :raise
            ConnectionError (Exception):
        '''
        try:
            # each request can retrieve up to 300 data points
            res = self.rest_client.get_product_historic_rates(name, granularity=grans)
            assert len(res) == 300, 'Length error!'
        except Exception as e:
            raise ConnectionError(f'cannot get historic rates for {name}')

        # WARNING: must flip order!
        res = sorted(res, key=lambda x: x[0], reverse=False)

        # want formatted datetime for clearer presentation
        for index,item in enumerate(res):
            '''Each item looks like: [
                [ time, low, high, open, close, volume ],
                [ 1415398768, 0.32, 4.2, 0.35, 4.2, 12.3 ], 
                    ...
                ]
            '''
            fmt_dt_str = datetime.datetime.fromtimestamp(item[0]).strftime('%Y-%m-%d %H:%M:%S').split(' ')[0]
            # grab desired variables
            open_price, closing_price = item[3], item[4]
            low, high = item[1], item[2]
            # append to the final list
            res[index] = [closing_price, fmt_dt_str, open_price, low, high]

        # today as a string
        today_str = datetime.datetime.now().strftime('%Y-%m-%d')
        # exclude today's data
        res = list(filter(lambda x: x[1] != today_str, res))

        return res

    @property
    def portfolio_value(self):
        '''Computes portfolio value for an account using Advanced Trade API.'''
        wallets = self.get_wallets()
        self.logger.debug(f'wallets: {wallets}')
        v_crypto, v_stable = 0, 0
        for item in wallets:
            try:
                delta_v = float(item['available_balance']['value'])
                if item['type'] == 'ACCOUNT_TYPE_CRYPTO':
                    v_crypto += delta_v
                elif item['currency'] in STABLECOIN:
                    v_stable += delta_v
            except Exception as e:
                self.logger.error(f'Exception processing wallet item: {item}, error: {e}')
                continue
        return np.round(v_crypto, 2), np.round(v_stable, 2)

    def get_wallets(self, cur_names: List[str]=[*CURS, *STABLECOIN]):
        '''Retrieve wallet information for pre-defined currency names using Advanced Trade API.'''
        try:
            accounts = self.rest_client.get_accounts()
            wallets = [x for x in accounts.accounts if x['currency'] in cur_names]
            self.logger.debug(f'get_accounts response: {wallets}')
            return wallets
        except Exception as e:
            self.logger.error(f'Exception when calling get_accounts: {e}')
            return []

    def place_buy_order(self, wallet_id: str, amount: float, currency: str, commit: bool = False):
        '''Place and (optionally) execute a buy order.

        :argument
            wallet_id (str):
            amount (float):
            currency (str):
            commit (boolean): Whether to commit this transaction.

        :return
            order (dictionary):

        :raise
            (Exception):
        '''
        try:
            order = self.rest_client.buy(wallet_id,
                        amount=str(amount),
                        currency=currency,
                        commit=commit
                    )
        except Exception as e:
            raise e

        return order

    def place_sell_order(self, wallet_id: str, amount: float, currency: str, commit: bool = False):
        '''Place and (optionally) execute a sell order.

        :argument
            wallet_id (str):
            amount (float):
            currency (str):
            commit (boolean): Whether to commit this transaction.

        :return
            order (dictionary):

        :raise
            (Exception):
        '''
        try:
            order = self.rest_client.sell(wallet_id,
                         amount=str(amount),
                         currency=currency,
                         commit=commit
                )
        except Exception as e:
            raise e

        return order