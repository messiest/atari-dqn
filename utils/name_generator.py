import requests


def fetch_name():
    URL = r"https://frightanic.com/goodies_content/docker-names.php"
    r = requests.get(URL)

    return r.text


if __name__ == "__main__":
    name = fetch_name()
    print(name)
