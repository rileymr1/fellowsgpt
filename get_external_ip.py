import requests

def get_external_ip():
    response = requests.get("https://api64.ipify.org?format=json")
    if response.status_code == 200:
        data = response.json()
        return data.get("ip")
    else:
        return "Unknown"