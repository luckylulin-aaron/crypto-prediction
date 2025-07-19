# built-in packages
import sys
import unittest

# customized packages
sys.path.insert(0, "../../app")

from app.core.config import *
from app.trading.trader_driver import TraderDriver
from app.utils.util import calculate_simulation_amounts, load_csv


class TestMADriver(unittest.TestCase):
    def test_regular_load_from_local(self):
        """Test loading data stream from local file."""

        name, l = load_csv(csv_fn="tests/fixtures/BTC_HISTORY.csv")
        l = l[-120:]
        # Expand to (price, date, open, low, high)
        l = [(row[0], row[1], row[0], row[0], row[0]) for row in l]

        # print('debug:', type(l[0][0]), type(l[0][1]), type(l[0]))

        # Use simulation amounts for testing
        sim_cash, sim_coin = calculate_simulation_amounts(
            actual_cash=3000,
            actual_coin=5,
            method="FIXED",  # Use fixed for testing
            base_amount=10000,
            percentage=0.1,
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


if __name__ == "__main__":
    unittest.main()
