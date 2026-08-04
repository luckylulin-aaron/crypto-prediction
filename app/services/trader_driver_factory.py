"""Factory for consistently configured stock and crypto TraderDriver objects."""


class TraderDriverFactory:
    def __init__(self, driver_class, common_parameters):
        self._driver_class = driver_class
        self._common_parameters = dict(common_parameters)

    def create(
        self,
        *,
        name,
        initial_cash,
        initial_coin,
        strategies,
        buy_pcts,
        sell_pcts,
        btc_data_stream=None,
        **overrides,
    ):
        parameters = dict(self._common_parameters)
        parameters.update(overrides)
        return self._driver_class(
            name=name,
            init_amount=initial_cash,
            cur_coin=initial_coin,
            overall_stats=list(strategies),
            buy_pcts=buy_pcts,
            sell_pcts=sell_pcts,
            btc_data_stream=btc_data_stream,
            **parameters,
        )
