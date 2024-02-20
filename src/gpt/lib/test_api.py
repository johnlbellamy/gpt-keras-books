import requests

if __name__ == '__main__':
    json_obj = {"prompt": "the sea was green and angry"}
    url = "http://localhost:5600/generate"
    response = requests.post(url, json=json_obj)
    try:
        print(f"{response.json()}")
    except:
        print("Looks like there response was empty or an error occurred.")
