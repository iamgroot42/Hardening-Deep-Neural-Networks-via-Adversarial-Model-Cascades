import json
from keras.models import load_model
from keras.models import model_from_json

before = model_from_json(json.load(open("before")))
after = model_from_json(json.load(open("after")))


