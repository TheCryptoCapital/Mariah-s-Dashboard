import requests
import os

ETHERSCAN_API_KEY = os.getenv("ETHERSCAN_API_KEY")

def get_eth_gas():
    try:
        url = f"https://api.etherscan.io/api?module=gastracker&action=gasoracle&apikey={ETHERSCAN_API_KEY}"
        response = requests.get(url).json()
        result = response["result"]
        return {
            "low": result["SafeGasPrice"],
            "avg": result["ProposeGasPrice"],
            "high": result["FastGasPrice"]
        }
    except:
        return {}

def get_block_info():
    try:
        url = f"https://api.etherscan.io/api?module=proxy&action=eth_blockNumber&apikey={ETHERSCAN_API_KEY}"
        response = requests.get(url).json()
        block = int(response["result"], 16)
        return block
    except:
        return "-"
    return 19876543
