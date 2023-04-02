import requests
import time


def readfile(FileName, chunk_size=5242880):
    with open(FileName, 'rb') as f:
        while True:
            data = f.read(chunk_size)
            if not data:
                break
            yield data


def WaitForCompletion(polling_endpoint, Author):
    while True:
        polling_response = requests.get(polling_endpoint, headers=Author).json()

        if polling_response['status'] == 'completed':
            break

        time.sleep(5)