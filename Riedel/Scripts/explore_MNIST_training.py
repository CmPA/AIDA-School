import numpy as np
from keras.datasets import mnist

# download and shuffled as training and testing set
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# function to explore one hand-written character
def character_show(character):
    for y in character:
        row = ""
        for x in y:
            row += '{0: <4}'.format(x)
        print(row)  

# view first 10 hand-written characters
for i in range (0,50):
    character_show(X_test[i])
    print("\n")
    print("Label:")
    print(y_test[i])
    print("\n")

