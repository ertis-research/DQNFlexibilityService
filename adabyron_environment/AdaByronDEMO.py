import gymnasium as gym
from gymnasium import spaces

from .airConditioning import Airconditioning
from .chargingStation import ChargingStation
from .storageBattery import StorageBattery


class AdaByronDEMO(gym.Env):
    """UMA DEMOSITE SCENARIO"""
    metadata = {'render.modes': ['human', 'computer']}

    def __init__(self, airConditionings, chargingStations, batteries): # capacity = 120.0, soc = 0.5):
        super(AdaByronDEMO, self).__init__()
        # Initialize all components
        self.airConditionings_list = airConditionings
        self.chargingStations_list = chargingStations
        self.batteries_list = batteries  
        
        self.agents = []
        self.initializeAgents()
        
        total_actions = []
        
        for agent in self.agents:
            total_actions.append(agent.get_action_space())

        self.action_space = (spaces.Discrete(sum(total_actions)), total_actions)
        
        self.consumption = 0
        self.cumulative_consumption = 0
        self.reward = 0
        self._episode_ended = False
        
    def initializeAgents(self):
        for air in self.airConditionings_list:
            self.agents.append(Airconditioning(air))
        for cs in self.chargingStations_list:
            self.agents.append(ChargingStation(cs))
        for bat in self.batteries_list:
            self.agents.append(StorageBattery(bat))
        
    
    def reset(self):
        # Reset the state of the environment to an initial state
        for agent in self.agents:
            agent.reset()

        self.consumption = 0
        self.cumulative_consumption = 0
        self.reward = 0
        self._episode_ended = False
        
        return self.getState(), self.reward, self._episode_ended
        
    def render(self, mode='human', close=False):
        # Render the environment to the screen
        if mode == 'human':
            for ac in self.airConditionings:
                ac.render()
            for cs in self.chargingStations:
                cs.render()
            for sb in self.batteries:
                sb.render()
            print("Actual consumption: {}, Cumulative consumption: {}, ENDED: {}, Reward: {}".format(self.consumption, self.cumulative_consumption, self._episode_ended, self.reward))
    
    def getState(self):
        return self.consumption
    
    def getAgentsActionSpace(self):
        return [agent.get_action_space() for agent in self.agents]
    
    def step(self, action, indice, generated_energy):
        
        # The action space is an array that contains the action for each agent
        # Execute one time step within the environment
        # for agent in self.agents:
        #     agent.stop()
        if indice == len(generated_energy)-1:
            self._episode_ended = True
            
        c = 0
        # Execute the action for each agent
        # No esta preparado para los cargadores
        
        
        # for index in range(len(self.agents)):
        #     if isinstance(self.agents[index], ChargingStation):
        #         c+=self.agents[index].step(action[index][0], action[index][1])
        #     else:
        #         c+=self.agents[index].step(action[index])
        
        fixed_actions = []
        aux = 0
        for i in range(len(action)):
            fixed_actions.append(action[i]-aux)
            #print("La accion {} es {}, con aux {}".format(action[i], action[i]-aux, aux))
            aux += self.action_space[1][i]
            #print(self.action_space[1][i])
        
        print("Fixed actions: {}".format(fixed_actions))
        
        for index in range(len(self.agents)):
            print("Index : {}".format(index))
            c+=self.agents[index].step(fixed_actions[index])
            print("Consumption: {}".format(c))
        self.cumulative_consumption += c
        self.consumption = c
        
        self.reward = 0 if self.consumption - generated_energy[len(generated_energy)-1] == 1 else (0-abs(self.consumption - generated_energy[len(generated_energy)-1]))/100
        
        return self.getState(), self.reward, self._episode_ended