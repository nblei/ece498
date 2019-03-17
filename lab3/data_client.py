import requests
import gzip
import numpy as np
import os

def get_testset(teamname):
    url = 'https://courses.engr.illinois.edu/ece498icc/sp2019/lab2_request_dataset.php'
    values = {'request': 'testdata', 'netid':'nbleier3', 'teamname' : teamname}
    r = requests.post(url, data=values, allow_redirects=True)
    print(r.url)
    filename = r.url.split("/")[-1]
    testset_id = filename.split(".")[0].split("_")[-1]
    with open(filename, 'wb') as f:
        f.write(r.content)

    print(testset_id)
    return load_dataset(filename), testset_id

def post_prediction(pred, data_id, teamname, latency):
    assert(len(pred) == 1000)
    url = 'https://courses.engr.illinois.edu/ece498icc/sp2019/lab2_request_dataset.php'
    values = {'request' : 'verify', 'netid' : 'nbleier3', 'testset_id' :
              data_id, 'prediction' : pred, 'team' : teamname, 'latency' : latency}
    print(data_id)
    r = requests.post(url, data=values, allow_redirects=True)
    os.remove("images_{dataid}.gz".format(dataid=data_id))
    print(r.content)
    return r

def load_dataset(path):
    num_img = 1000
    with gzip.open(path, 'rb') as infile:
        data = np.frombuffer(infile.read(), dtype=np.uint8).reshape(num_img, 784)
        data = data / 255.0
        data.resize((num_img, 28, 28, 1))
    return data

