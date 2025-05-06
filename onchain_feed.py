import os
import requests
from dotenv import load_dotenv

load_dotenv()
ETHERSCAN_API_KEY = os.getenv("ETHERSCAN_API_KEY")

def get_eth_gas():
    url = f"https://api.etherscan.io/api?module=gastracker&action=gasoracle&apikey={ETHERSCAN_API_KEY}"
    try:
        res = requests.get(url)
        print("🔍 GAS API Response:", res.text)
        data = res.json()
        return {
            "low": round(float(data["result"]["SafeGasPrice"]), 2),
            "avg": round(float(data["result"]["ProposeGasPrice"]), 2),
            "high": round(float(data["result"]["FastGasPrice"]), 2),
            "timestamp": data["result"]["LastBlock"]
        }
    except Exception as e:
        print("❌ Failed to fetch gas price:", e)
        return {}

def get_block_info():
    url = f"https://api.etherscan.io/api?module=proxy&action=eth_blockNumber&apikey={ETHERSCAN_API_KEY}"
    try:
        res = requests.get(url)
        data = res.json()
        return int(data["result"], 16)
    except Exception as e:
        print("❌ Failed to fetch block info:", e)
        return None

