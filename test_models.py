from keras.models import load_model
import numpy as np
import random


if __name__ == '__main__':
    input_vector = np.array([[random.random() for _ in range(430)] for _ in range(1)])
    # print(input_vector.shape)
    # print(input_vector)
    input_vector = input_vector.reshape(input_vector.shape[0], 1, 430, 1)
    person_model = load_model("saved_models/person_model.h5")
    gender_model = load_model("saved_models/gender_model.h5")

    pred = person_model.predict(input_vector)
    print(pred)