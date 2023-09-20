import tensorflow as tf
import random
import numpy as np
from adabyron_environment.AdaByronDEMO import AdaByronDEMO
from .models import *
    
def define_env(run, aires = [], cargadores = [], baterias = [], INPUT_SIZE=0):
        
    env = AdaByronDEMO(aires, cargadores, baterias)
    action_space = env.action_space
    actionsPerAgents = env.getAgentsActionSpace()
    
    try:
        print("Loading models...")
        policy_network = tf.keras.models.load_model("models/{}/policyNetwork.h5".format(run))
        target_network = tf.keras.models.load_model("models/{}/targetNetwork.h5".format(run))
    except Exception as e:
        print("Models not found, creating new ones...")
        policy_network = multipleOutputs_DQN(INPUT_SIZE, actionsPerAgents)
        target_network = multipleOutputs_DQN(INPUT_SIZE, actionsPerAgents)
    
    return env, policy_network, target_network, action_space  
      
def train(epsilon, MAX_STEPS, action_space, policy_network, EPS_DECAY, EPS_MIN, env, memory, BATCH_SIZE, STEP_UPDATE_MODEL, target_network, K, gamma, loss_function, optimizer, generated_energy):
    step = 0
    while True:
        inicial_state_consumption, _, _ = env.reset()
        day_step = 0
        initial_state = np.array([inicial_state_consumption, day_step]+generated_energy)
        
        
        for i in range(1, MAX_STEPS):
            step+=1
            if step < 50000 or np.random.rand(1)[0] < epsilon:
                #action = random.randint(0, action_space[0]-1)
                action = [random.randint(0, agent_as) for agent_as in action_space[1]]
            else:
                tf_tensor = tf.convert_to_tensor(initial_state)
                tf_tensor = tf.expand_dims(initial_state, 0)

                output = policy_network(tf_tensor, training=False)
                print(output)
                action = [tf.argmax(i[0]).numpy() for i in output]
                #action = tf.argmax(output[0]).numpy()
            epsilon-=EPS_DECAY

            epsilon = max(epsilon, EPS_MIN)
            
            next_step, reward, done = env.step(action, day_step, generated_energy)
            #next_step = np.array(next_step)
            day_step+=1
            next_step = [next_step, day_step]+generated_energy
            if not done:
                #memory.save(initial_state, action, reward, next_step, float(done))
                memory.save(initial_state, action, reward, next_step, done)
                initial_state = next_step
            
            if step % K == 0 and len(memory) >= BATCH_SIZE:
                optimize_model(memory, BATCH_SIZE, target_network, policy_network, gamma, loss_function, optimizer, action_space)
            if step % STEP_UPDATE_MODEL == 0:
                target_network.set_weights(policy_network.get_weights())
                target_network.save("models/{}/targetNetwork_fineT.h5".format("test"))
                policy_network.save("models/{}/policyNetwork_fineT.h5".format("test"))
            if done: break
            
def optimize_model(memory, BATCH_SIZE, target_network, policy_network, gamma, loss_function, optimizer, action_space):
    states, actions, rewards, next_steps, dones = memory.sample(BATCH_SIZE)
    #dones_tensor = tf.convert_to_tensor(dones)
    #print(len(memory))
    print(len(next_steps))
    print(next_steps)
    tam_max = max(action_space[1])
    qpvalues = target_network.predict(next_steps, verbose=0)
    

    qpvalues_2 = []
    for v in range(len(qpvalues)):
        aux = tf.expand_dims(np.append(qpvalues[v][0], [float("-inf") for i in range(tam_max-action_space[1][v])]), 0)
        qpvalues_2.append(aux)
        #qpvalues[v][0]=np.concatenate((qpvalues[v][0],[float("-inf") for i in range(tam_max-action_space[1][v])]), axis=None)

    
    maximos = tf.reduce_max(qpvalues_2, axis = 2)
    y_target = rewards + gamma*maximos
    
    # Comento esta línea porque no existen los estados finales negativos en este problema
    #y_target = y_target * (1-dones_tensor) - dones_tensor

    mask = []
    
    for i in range(len(actions[0])):
        mask.append(tf.expand_dims(tf.one_hot(actions[0][i], action_space[1][i]), 0))
    #mask = tf.one_hot(actions, action_space[1])
    print("Mask")
    print(mask)
    
    with tf.GradientTape() as cinta:
        Qvalues = policy_network(states)
        print("Qvalues")
        print(Qvalues)
        print(type(Qvalues))
        print(type(mask))
        
        for i in range(len(Qvalues)):
            tf.multiply(Qvalues[i], mask[i])
            
        mult = tf.multiply(Qvalues, mask)
        y_pred = tf.reduce_sum(mult)
        loss = loss_function(y_target, y_pred)

    gradients = cinta.gradient(loss, policy_network.trainable_variables)
    optimizer.apply_gradients(zip(gradients, policy_network.trainable_variables))