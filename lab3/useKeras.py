import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import time
from tensorflow import keras
from  kerasModel import make_model
from data_client import get_testset, post_prediction

TEAMNAME = "Team Sparks"

model = make_model()
model.load_weights("./keras_model")

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


data, data_id = get_testset(TEAMNAME)

start = time.time()
preds = model.predict(data)

preds = list(map(np.argmax, preds))

result = "".join(map(str, preds))
end = time.time()

print("Inference time {sec} seconds".format(sec=end-start))

r = post_prediction(result, data_id, TEAMNAME, end-start)
print(r)

