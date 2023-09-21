import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras

def basic_DQN(INPUT_SIZE, NUM_ACTIONS, capaEntrada):
    # Network defined by the Deepmind paper
    # inputs = layers.Input(shape=(1, 10))
    inputs = capaEntrada

    # Hidden layers
    layer1 = layers.Dense(128, activation="relu")(inputs)
    layer2 = layers.Dense(256, activation="relu")(layer1)
    layer3 = layers.Dense(256, activation="relu")(layer2)
    layer4 = layers.Dense(512, activation="relu")(layer3)
    action = layers.Dense(NUM_ACTIONS, activation="linear")(layer4)

    return keras.Model(inputs=inputs, outputs=action)


def multipleOutputs_DQN(INPUT_SIZE, num_actions):
    # Network defined by the Deepmind paper
    inputs = layers.Input(shape=(INPUT_SIZE, ))

    # Hidden layers
    layer1 = layers.Dense(128, activation="relu")(inputs)
    layer2 = layers.Dense(256, activation="relu")(layer1)
    layer3 = layers.Dense(256, activation="relu")(layer2)
    layer4 = layers.Dense(512, activation="relu")(layer3)
    
    output = layers.Dense(num_actions, activation="linear")(layer4)
    #capas = [layers.Dense(actions, activation="linear")(layer4) for actions in actionsPerAgents]


    return keras.Model(inputs=inputs, outputs=output)


def manDQN(INPUT_SIZE, actionsPerAgents):
    inputs = layers.Input(shape=(INPUT_SIZE, ))
    
    redes = [basic_DQN(INPUT_SIZE, actions, inputs) for actions in actionsPerAgents]
    
    return keras.Model(inputs=inputs, outputs=redes)