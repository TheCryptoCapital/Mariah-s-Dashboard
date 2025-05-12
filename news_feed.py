import requests

def get_crypto_news():
    try:
        url = "https://cryptopanic.com/api/v1/posts/?auth_token=demo&public=true"
        response = requests.get(url)
        articles = response.json().get("results", [])
        return articles
    except Exception as e:
        return []
