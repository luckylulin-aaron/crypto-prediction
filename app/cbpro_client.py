import cbpro
import datetime

from coinbase.wallet.client import Client
from typing import List, Any

from config import *
from util import *

class CBProClient:

    def __init__(self, key: str='', secret: str='', pwd: str=''):
        # by default will have a public client
        self.public_client = cbpro.PublicClient()
        # whether to create an authenticated client
        if len(key) > 0 and len(secret) > 0:
            self.auth_client = Client(key, secret)
        else:
            self.auth_client = None

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
            res = self.public_client.get_product_24hr_stats(name)['last']
        except Exception as e:
            raise ConnectionError(f'cannot get current rate for {name}')

        res = float(res)
        return res

    @timer
    def get_historic_data(self, name: str, grans: int):
        '''Get historic rates.

        :argument
            name (str): Currency name.
            grans (int): Granularity, i.e. desired timeslice in seconds.

        :return
            res (List[List[float, str]]): Data stream of prices and dates.

        :raise
            ConnectionError (Exception):
        '''
        try:
            # each request can retrieve up to 300 data points
            res = self.public_client.get_product_historic_rates(name, granularity=grans)
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
            closing_price = item[3]
            res[index] = [closing_price, fmt_dt_str]

        # today as a string
        today_str = datetime.datetime.now().strftime('%Y-%m-%d')
        # exclude today's data
        res = list(filter(lambda x: x[-1] != today_str, res))

        return res

    def get_wallets(self, cur_names: List[str]=[*CURS, *FIAT]):
        '''Retrieve wallet information for pre-defined currency names.
        Those with type being vault will be dropped.

        :argument
            cur_names （List[str]):

        :return
            List[dictionary]:

        :raise
        '''
        accts = self.auth_client.get_accounts()

        return list(filter(lambda x: x['currency'] in cur_names and \
                                     x['type'] != 'vault', accts['data']))

    def place_buy_order(self, wallet_id: str, amount: float, currency: str, commit: bool=False):
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
            order = self.auth_client.buy(wallet_id,
                        amount=str(amount),
                        currency=currency,
                        commit=commit
                    )
        except Exception as e:
            raise e

        return order

    def place_sell_order(self, wallet_id: str, amount: float, currency: str, commit: bool=False):
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
            order = self.auth_client.sell(wallet_id,
                         amount=str(amount),
                         currency=currency,
                         commit=commit
                )
        except Exception as e:
            raise e

        return order

    @property
    def portfolio_value(self):
        '''Computes portfolio value for an account.

        :argument

        :return
            v_crypto (float): Value of all crypto-currencies in USD.
            v_fiat (float): Value of fiat currency in USD.

        :raise
        '''
        assert self.auth_client is not None, 'no authorized account to get information from!'
        wallets = self.get_wallets()

        v_crypto, v_fiat = 0, 0

        for item in wallets:
            delta_v = float(item['native_balance']['amount'])
            # add to different variables
            if item['type'] == 'wallet':
                v_crypto += delta_v
            elif item['type'] == 'fiat':
                v_fiat += delta_v

        return v_crypto, v_fiat