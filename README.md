# crypto-prediction

Integration of some simulation models and automatic trading via coinbase API

## Warning
To perform actual transaction, you need to have an existing coinbase account and enable its API.

## Installation

1) Clone this repository.
2) In the repository, execute `pip install -r requirements.txt`
3) Manually change your API key and API secret in ```app/credentials.py```, and tweak other parameters of your choice in the same file. Note: credentials.py should contain only two variables:  CB_API_KEY = '...'; CB_API_SECRET = '...'.

## Testing
Navigate to ```tests/unit```, execute `python -m unittest` to launch unittest that reads sample data locally.

## Reference
I have no intent to commercialize this project. Sincere thanks to programmers around the world who make it possible. Refer to the following sources if you have technical problems.

1) https://github.com/coinbase/coinbase-python
2) https://developers.coinbase.com/api/v2?python#2017-08-07 