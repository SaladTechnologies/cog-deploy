import requests
from time import sleep

url = 'http://[::1]:5000/health-check'

while True:
    sleep(1)
    print("---------------------------------->")

    try:
        x = requests.get(url)
        print(x.status_code)
        print(x.content)

    except Exception as e:
        print(e)
