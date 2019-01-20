import os
import pickle
import numpy as np
import tensorflow as tf
from tensorflow import keras

def create_network():
    model = keras.Sequential([
        keras.layers.Dense(64, activation=tf.nn.relu, input_shape=(17,)),
        keras.layers.Dense(64, activation=tf.nn.relu),
        keras.layers.Dense(6)
    ])

    model.compile(optimizer=tf.train.AdamOptimizer(),
                  loss='mse',
                  metrics=['accuracy'])

    return model

def load_expert_data():
    with open(os.path.join("expert_data", "HalfCheetah-v2" + ".pkl"), "rb") as f:
        expert_data = pickle.load(f)
        return expert_data["observations"], expert_data["actions"]

def train(model, train_datas, train_labels, test_datas, test_labels):
    model.fit(train_datas, train_labels, epochs=100, validation_data=(test_datas, test_labels))

def main():
    obs, act = load_expert_data()
    print(obs[0].shape)
    print(act[0].shape)
    train_datas = obs[:40000]
    train_labels = np.reshape(act[:40000], (40000, 6))
    test_datas = obs[40000:]
    test_labels = np.reshape(act[40000:], (10000, 6))
    
    # print(train_labels[:2])
    model = create_network()
    train(model, train_datas, train_labels, test_datas, test_labels)
    test_loss, test_acc = model.evaluate(test_datas, test_labels)
    model.save(os.path.join("model", "HalfCheetah-v2-model-40000-d64-d64-e100.h5"))
    print(test_loss, test_acc)

if __name__ == '__main__':
    main()