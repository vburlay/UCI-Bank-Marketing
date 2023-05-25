import matplotlib.pyplot as plt
import numpy as np
from keras.models import Sequential
import keras
from keras.layers import Dense
from keras.layers import Input, Flatten, Dropout, Activation
from keras.layers import Conv1D, MaxPooling1D, AveragePooling1D
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D,SpatialDropout1D
from keras.models import Model

def build_model():
    model = Sequential()

    model.add(Conv1D(128, 5,padding='same',
                 input_shape=(63,1)))
    model.add(Activation('relu'))
    model.add(Dropout(0.1))
    model.add(MaxPooling1D(pool_size=(8)))
    model.add(Conv1D(128, 5,padding='same',))
    model.add(Activation('relu'))
    model.add(Dropout(0.1))
    model.add(Flatten())
    model.add(Dense(2)) # auf 2 setzen und neu trainieren
    model.add(Activation('softmax'))
    return model
def k_fold(k,x_traincnn,x_testcnn,y_train,y_test):
    num_val_samples = len(x_traincnn) // k
    num_epoches = 100
    opt = keras.optimizers.RMSprop(learning_rate=0.00005, rho=0.9, epsilon=None, decay=0.0)
    for i in range(k):
        print(f"Processing fold #{i}")
        val_data = x_traincnn[i * num_val_samples: (i + 1) * num_val_samples]
        val_targets = y_train[i * num_val_samples: (i + 1) * num_val_samples]
        partial_train_data = np.concatenate(
        [x_traincnn[:i * num_val_samples],
        x_traincnn[(i + 1) * num_val_samples:]],
        axis = 0)
        partial_train_targets = np.concatenate(
        [y_train[:i * num_val_samples],
        y_train[(i + 1) * num_val_samples:]],
        axis = 0)
        model = build_model()
        model.compile(loss='sparse_categorical_crossentropy',
              optimizer=opt,
              metrics=["acc"])
        cnnhistory=model.fit(partial_train_data, partial_train_targets, batch_size=16, epochs=num_epoches, verbose = 0)
        loss, acc = model.evaluate(val_data, val_targets, verbose = 0)
        print("Restored model, accuracy: {:5.2f}%".format(100*acc),f"loss:{loss}")
    loss, acc = model.evaluate(x_testcnn, y_test)
    print("Restored model, accuracy: {:5.2f}%".format(100*acc))
    return model,cnnhistory 

