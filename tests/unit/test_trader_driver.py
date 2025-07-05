# built-in packages
import configparser
import datetime
import os
import sys
import unittest

# third-party packages

# customized packages
sys.path.insert(0, "../../app")

from cbpro_client import CBProClient
from config import *
from trader_driver import TraderDriver
from util import (load_csv)


class TestMADriver(unittest.TestCase):
    def test_regular_load_from_local(self):
        """Test loading data stream from local file."""

        name, l = load_csv(csv_fn="tests/fixtures/BTC_HISTORY.csv")
        l = l[-120:]
        # Expand to (price, date, open, low, high)
        l = [(row[0], row[1], row[0], row[0], row[0]) for row in l]

        # print('debug:', type(l[0][0]), type(l[0][1]), type(l[0]))

        # Use simulation amounts for testing
        from util import calculate_simulation_amounts
        
        sim_cash, sim_coin = calculate_simulation_amounts(
            actual_cash=3000,
            actual_coin=5,
            method="FIXED",  # Use fixed for testing
            base_amount=10000,
            percentage=0.1
        )

        t_driver = TraderDriver(
            name=name,
            init_amount=int(sim_cash),
            overall_stats=STRATEGIES,
            cur_coin=sim_coin,
            tol_pcts=TOL_PCTS,
            ma_lengths=MA_LENGTHS,
            ema_lengths=EMA_LENGTHS,
            bollinger_mas=BOLLINGER_MAS,
            bollinger_tols=BOLLINGER_TOLS,
            buy_pcts=BUY_PCTS,
            sell_pcts=SELL_PCTS,
            buy_stas=BUY_STAS,
            sell_stas=SELL_STAS,
            mode="normal",
        )

        t_driver.feed_data(data_stream=l)

        info = t_driver.best_trader_info
        # best trader
        best_t = t_driver.traders[info["trader_index"]]
        # for a new price, find the trade signal
        signal = best_t.trade_signal

        # check
        print("examine best trader performance:", info)
        self.assertNotEqual(info["trader_index"], None), "incorrect best trader index!"

    def test_run_with_cbpro(self):
        """Test running with CBProClient that requires internet connection."""
        return
        
        # Read API credentials from screte.ini
        config = configparser.ConfigParser()
        config.read(os.path.join(os.path.dirname(__file__), "../../app/screte.ini"))
        
        CB_API_KEY = config["CONFIG"]["COINBASE_API_KEY"].strip('"')
        CB_API_SECRET = config["CONFIG"]["COINBASE_API_SECRET"].strip('"')
        
        client = CBProClient(key=CB_API_KEY, secret=CB_API_SECRET)

        for cur_name in CURS:
            r1 = client.get_cur_rate(cur_name + "-USD")
            data_stream = client.get_historic_data(
                cur_name + "-USD", SECONDS_IN_ONE_DAY
            )

            # Use simulation amounts for testing
            sim_cash, sim_coin = calculate_simulation_amounts(
                actual_cash=3000,
                actual_coin=5,
                method="FIXED",  # Use fixed for testing
                base_amount=10000,
                percentage=0.1
            )

            t_driver = TraderDriver(
                name=cur_name,
                init_amount=int(sim_cash),
                overall_stats=STRATEGIES,
                cur_coin=sim_coin,
                tol_pcts=TOL_PCTS,
                ma_lengths=MA_LENGTHS,
                ema_lengths=EMA_LENGTHS,
                bollinger_mas=BOLLINGER_MAS,
                bollinger_tols=BOLLINGER_TOLS,
                buy_pcts=BUY_PCTS,
                sell_pcts=SELL_PCTS,
                buy_stas=BUY_STAS,
                sell_stas=SELL_STAS,
                mode="normal",
            )

            t_driver.feed_data(data_stream=data_stream)

            info = t_driver.best_trader_info
            # best trader
            best_t = t_driver.traders[info["trader_index"]]

            # for a new price, find the trade signal
            new_p = client.get_cur_rate(cur_name + "-USD")
            today = datetime.datetime.now().strftime("%m/%d/%Y")
            # add it and compute
            best_t.add_new_day(new_p, today)
            signal = best_t.trade_signal

            # check
            self.assertEqual(best_t.crypto_name, cur_name), "incorrect crypto name!"
            self.assertIsInstance(
                signal, dict
            ), "trading signal must be of type dictionary!"
            print("examine best trader performance:", info)
            print(f"for crypto={best_t.crypto_name}, today's signal={signal}")


if __name__ == "__main__":
    unittest.main()
