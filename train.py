from utils import *
from network import *
import pandas as pd
import sys
import json

model = network()
while True:
    jsonString = sys.stdin.readline()
    dict = json.loads(jsonString)

    x_train = np.array(dict['x_train'])
    x_val = np.array(dict['x_val'])
    x_test = np.array(dict['x_test'])
    y_train = np.array(dict['y_train'])
    y_val = np.array(dict['y_val'])
    y_test = np.array(dict['y_test'])

    history = model.fit(x = x_train, y = y_train, validation_data = (x_val, y_val),
                     batch_size=64,
                      epochs=10)
    # Calculate its accuracy on testing data
    _,acc = model.evaluate(x_val, y_val)

    print('The accuracy on the validation data is {}%.'.format(acc*100))
    # Save the model
    model.save('model1_10epoch.h5')

