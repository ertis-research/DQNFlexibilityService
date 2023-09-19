import tensorflow as tf
import random
import numpy as np
from adabyron_environment.AdaByronDEMO import AdaByronDEMO
from .models import *
    
def define_env(run, aires = [], cargadores = [], baterias = [], INPUT_SIZE=0):
        
    env = AdaByronDEMO(aires, cargadores, baterias)
    num_actions = env.action_space
    actionsPerAgents = env.getAgentsActionSpace()
    
    try:
        print("Loading models...")
        policy_network = tf.keras.models.load_model("models/{}/policyNetwork.h5".format(run))
        target_network = tf.keras.models.load_model("models/{}/targetNetwork.h5".format(run))
    except Exception as e:
        print("Models not found, creating new ones...")
        policy_network = multipleOutputs_DQN(INPUT_SIZE, actionsPerAgents)
        target_network = multipleOutputs_DQN(INPUT_SIZE, actionsPerAgents)
    
    return env, policy_network, target_network, num_actions  
      
def train(epsilon, MAX_STEPS, num_actions, policy_network, EPS_DECAY, EPS_MIN, env, memory, BATCH_SIZE, STEP_UPDATE_MODEL, target_network, K, gamma, loss_function, optimizer, generated_energy):
    step = 0
    while True:
        inicial_state_consumption, _, _ = env.reset()
        initial_state = np.array([inicial_state_consumption, step]+generated_energy)
        
        
        for i in range(1, MAX_STEPS):
            step+=1
            if step < 50000 or np.random.rand(1)[0] < epsilon:
                action = random.randint(0, num_actions-1)
            else:
                tf_tensor = tf.convert_to_tensor(initial_state)
                tf_tensor = tf.expand_dims(initial_state, 0)

                output = policy_network(tf_tensor, training=False)
                action = [tf.argmax(i[0]).numpy() for i in output]
                #action = tf.argmax(output[0]).numpy()
            epsilon-=EPS_DECAY

            epsilon = max(epsilon, EPS_MIN)
            next_step, reward, done, _ = env.step(action, generated_energy[step])
            next_step = np.array(next_step)

            memory.save(initial_state, action, reward, next_step, float(done))
            initial_state = next_step

            if step % K == 0 and len(memory) >= BATCH_SIZE:
                optimize_model(memory, BATCH_SIZE, target_network, policy_network, gamma, loss_function, optimizer, num_actions)
            if step % STEP_UPDATE_MODEL == 0:
                target_network.set_weights(policy_network.get_weights())
                target_network.save("models/{}/targetNetwork_fineT.h5".format("test"))
                policy_network.save("models/{}/policyNetwork_fineT.h5".format("test"))
            if done: break
            
def optimize_model(memory, BATCH_SIZE, target_network, policy_network, gamma, loss_function, optimizer, num_actions):
    states, actions, rewards, next_steps, dones = memory.sample(BATCH_SIZE)
    dones_tensor = tf.convert_to_tensor(dones)

    qpvalues = target_network.predict(next_steps, verbose=0)
    y_target = rewards + gamma*tf.reduce_max(qpvalues, axis=1)
    y_target = y_target * (1-dones_tensor) - dones_tensor
    
    mask = tf.one_hot(actions, num_actions)
    
    with tf.GradientTape() as cinta:
        Qvalues = policy_network(states)
        
        y_pred = tf.reduce_sum(tf.multiply(Qvalues, mask), axis = 1)
        loss = loss_function(y_target, y_pred)

    gradients = cinta.gradient(loss, policy_network.trainable_variables)
    optimizer.apply_gradients(zip(gradients, policy_network.trainable_variables))