import datetime
import json
import numpy as np
import os
import pandas as pd

from cbpro_client import CBProClient
from trader_driver import TraderDriver
from config import *
from util import load_csv


def main():
    '''Test cbpro functionality.'''

    client = CBProClient(
        key=CB_API_KEY,
        secret=CB_API_SECRET
    )

    wallets = client.get_wallets(cur_names=[*CURS, *REAL_CURS])

    # place a sell order, not to commit nonetheless
    '''
    sell_order = client.auth_client.sell(
        '5beac516-8f09-5de4-937c-e497a74982ac',
        amount='1',
        currency='LTC',
        commit=False
    )
    '''
    buy_order = client.auth_client.buy(
        '5beac516-8f09-5de4-937c-e497a74982ac',
        amount='10',
        currency='USD',
        commit=False
    )

    with open('../tests/fixtures/buy_order_sample.json', 'w') as outfile:
        json.dump(buy_order, outfile, indent=4, ensure_ascii=False)


if __name__ == '__main__':

    main()