import requests
import os

def get_crypto_news():
    api_key = os.getenv("CRYPTOPANIC_API_KEY")
    url = f"https://cryptopanic.com/api/v1/posts/?auth_token={api_key}&public=true"

    try:
        response = requests.get(url)
        data = response.json()
        news = []

        for item in data.get("results", []):
            news.append({
                "title": item.get("title", "No title"),
                "url": item.get("url", ""),
                "published": item.get("published_at", ""),
                "source": item.get("source", {}).get("title", ""),
                "tags": item.get("currencies", []),
            })

        return news[:10]

    except Exception as e:
        print("❌ Failed to fetch crypto news:", e)
        return []

