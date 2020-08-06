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
        except Exception as e:
            raise ConnectionError(f'cannot get historic rates for {name}')


        # want formatted datetime for clearer presentation
        for index,item in enumerate(res):
            '''[
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

    @timer
    def get_wallets(self, cur_names: List[str]):
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