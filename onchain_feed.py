"""
On-Chain Data Feed Module
Handles ETH gas data and basic blockchain info
"""

import requests
import os
from datetime import datetime, timedelta

def get_eth_gas():
    """Get current ETH gas prices from Etherscan API"""
    try:
        api_key = os.getenv("ETHERSCAN_API_KEY")
        url = "https://api.etherscan.io/api"
        params = {
            "module": "gastracker",
            "action": "gasoracle",
            "apikey": api_key
        }
        
        response = requests.get(url, params=params, timeout=10)
        data = response.json()
        
        if data["status"] == "1":
            result = data["result"]
            return {
                "low": int(result["SafeGasPrice"]),
                "avg": int(result["StandardGasPrice"]), 
                "high": int(result["FastGasPrice"])
            }
        else:
            # Fallback values if API fails
            return {"low": 20, "avg": 25, "high": 30}
            
    except Exception as e:
        print(f"Error fetching gas data: {e}")
        # Return fallback values
        return {"low": 20, "avg": 25, "high": 30}

def get_block_info():
    """Get latest ETH block number"""
    try:
        api_key = os.getenv("ETHERSCAN_API_KEY")
        url = "https://api.etherscan.io/api"
        params = {
            "module": "proxy",
            "action": "eth_blockNumber",
            "apikey": api_key
        }
        
        response = requests.get(url, params=params, timeout=10)
        data = response.json()
        
        if "result" in data:
            # Convert hex to decimal
            block_number = int(data["result"], 16)
            return block_number
        else:
            return "Unknown"
            
    except Exception as e:
        print(f"Error fetching block info: {e}")
        return "Unknown"

def get_network_stats():
    """Get additional network statistics"""
    try:
        # Example: Get network congestion based on pending transactions
        # This would require additional API calls or different endpoints
        return {
            "congestion": "Medium",
            "avg_block_time": "12-15 seconds",
            "pending_txs": "~50,000"
        }
    except Exception as e:
        print(f"Error fetching network stats: {e}")
        return {
            "congestion": "Unknown",
            "avg_block_time": "Unknown", 
            "pending_txs": "Unknown"
        }