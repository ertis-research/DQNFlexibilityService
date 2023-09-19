import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras

def basic_DQN(NUM_ACTIONS):
    # Network defined by the Deepmind paper
    # inputs = layers.Input(shape=(1, 10))
    inputs = layers.Input(shape=(10,))

    # Hidden layers
    layer1 = layers.Dense(128, activation="relu")(inputs)
    layer2 = layers.Dense(256, activation="relu")(layer1)
    layer3 = layers.Dense(256, activation="relu")(layer2)
    layer4 = layers.Dense(512, activation="relu")(layer3)
    action = layers.Dense(NUM_ACTIONS, activation="linear")(layer4)

    return keras.Model(inputs=inputs, outputs=action)


def multipleOutputs_DQN(INPUT_SIZE, actionsPerAgents):
    # Network defined by the Deepmind paper
    inputs = layers.Input(shape=(INPUT_SIZE, ))

    # Hidden layers
    layer1 = layers.Dense(128, activation="relu")(inputs)
    layer2 = layers.Dense(256, activation="relu")(layer1)
    layer3 = layers.Dense(256, activation="relu")(layer2)
    layer4 = layers.Dense(512, activation="relu")(layer3)
    
    
    capas = [layers.Dense(actions, activation="linear")(layer4) for actions in actionsPerAgents]


    return keras.Model(inputs=inputs, outputs=capas)