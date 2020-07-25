from utils import *
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.optimizers import SGD


def network():
    number_classes = 7
    model = Sequential()
    model.add(Dense(units=128, input_shape=(262,), activation='relu'))
    model.add(Dropout(rate=0.2))
    model.add(Dense(units=64, activation='relu'))
    model.add(Dropout(rate=0.2))
    model.add(Dense(units=number_classes, activation='softmax'))

    sgd = SGD(lr=0.003, momentum=0.9, decay=(0.01 / 25), nesterov=False)
    model.compile(optimizer=sgd,
                  loss="categorical_crossentropy",
                  metrics=['accuracy'])

    return model