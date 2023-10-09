import tensorflow as tf
import random
import numpy as np
from adabyron_environment.AdaByronDEMO import AdaByronDEMO
from .models import *
    
def define_env(run, aires = [], cargadores = [], baterias = [], INPUT_SIZE=0):
        
    env = AdaByronDEMO(aires, cargadores, baterias)
    action_space = env.action_space
    
    try:
        print("Loading models...")
        policy_network = tf.keras.models.load_model("models/{}/policyNetwork_fineT.h5".format(run))
        target_network = tf.keras.models.load_model("models/{}/targetNetwork_fineT.h5".format(run))
    except Exception as e:
        print("Models not found, creating new ones...")
        policy_network = multipleOutputs_DQN(INPUT_SIZE, action_space[0].n)
        target_network = multipleOutputs_DQN(INPUT_SIZE, action_space[0].n)
    
    return env, policy_network, target_network, action_space  

def action_selection(output, action_space):
    output = output[0]
    action = []
    aux = 0
    for i in range(len(action_space[1])):
        action.append(tf.argmax(output[aux:action_space[1][i]+aux]).numpy()+aux)
        aux+=action_space[1][i]
    return action

def randomActions(action_space):
    aux = 0
    action = []
    for agent_actions in action_space[1]:
        action.append(random.randint(0, agent_actions-1)+aux)
        aux+=agent_actions
    return action


def train(epsilon, MAX_STEPS, action_space, policy_network, EPS_DECAY, EPS_MIN, env, memory, BATCH_SIZE, STEP_UPDATE_MODEL, target_network, K, gamma, loss_function, optimizer, generated_energy, episodes, run_name):
    step = 0
    episode = 0
    while episode <= episodes:
        if episode % 100 == 0:
            print("Episode: {}".format(episode))
        episode+=1
        inicial_state_consumption, _, _ = env.reset()
        day_step = 0
        initial_state = np.array([inicial_state_consumption, day_step]+generated_energy)
        
        
        for i in range(1, MAX_STEPS):
            step+=1
            if step < 50000 or np.random.rand(1)[0] < epsilon:
                #action = random.randint(0, action_space[0]-1)
                action = randomActions(action_space)
            else:
                tf_tensor = tf.convert_to_tensor(initial_state)
                tf_tensor = tf.expand_dims(initial_state, 0)

                output = policy_network(tf_tensor, training=False)
                action = action_selection(output, action_space)
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
                target_network.save("models/{}/targetNetwork_fineT.h5".format(run_name))
                policy_network.save("models/{}/policyNetwork_fineT.h5".format(run_name))
                print("Models saved")
                print("Epsilon {}".format(epsilon))
            if done: break
            
def optimize_model(memory, BATCH_SIZE, target_network, policy_network, gamma, loss_function, optimizer, action_space):
    states, actions, rewards, next_steps, dones = memory.sample(BATCH_SIZE)
 
    qpvalues = target_network.predict(next_steps, verbose=0)

    action_mask = action_selection(qpvalues, action_space)

    qpvalues_mask = tf.one_hot([action_mask], action_space[0].n, axis=2).numpy()
    qpvalues_mask = tf.reduce_sum(qpvalues_mask, axis=1).numpy()
    qpvalues = tf.multiply(qpvalues, qpvalues_mask)
    
    #qpvalues = tf.reduce_max(qpvalues, axis=1)
    qpvalues = tf.reduce_sum(qpvalues, axis=1)
    
    y_target = rewards + gamma*qpvalues
    # print("y_target")
    # print(y_target)
    # Comento esta línea porque no existen los estados finales negativos en este problema
    #y_target = y_target * (1-dones_tensor) - dones_tensor

    
    
    mask = tf.one_hot(actions, action_space[0].n, axis=2).numpy()
    mask = tf.reduce_sum(mask, axis=1).numpy()


    with tf.GradientTape() as cinta:
        Qvalues = policy_network(states)
        mult = tf.multiply(Qvalues, mask)
        y_pred = tf.reduce_sum(mult, axis = 1)
        loss = loss_function(y_target, y_pred)
        
        # print("y_pred")
        # print(y_pred)
        # print(y_target)

    gradients = cinta.gradient(loss, policy_network.trainable_variables)
    optimizer.apply_gradients(zip(gradients, policy_network.trainable_variables))