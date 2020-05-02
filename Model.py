import os
from keras.models import load_model
import Dataset
import CNN


def train():
    X_train, X_test, y_train_a,\
    y_train_g, y_test_a, y_test_g = Dataset.load_dataset()
    model = CNN.net()
    hist = model.fit(X_train, [y_train_g, y_train_a],
                     batch_size=32, epochs=3,
                     validation_data=(X_test, [y_test_g, y_test_a]))
    model.save('agender.h5')
    return model


def load():
    if os.path.exists('agender.h5'):
        model = load_model('agender.h5')
        return model
    else:
        return train()


if __name__=='__main__':
    load()
