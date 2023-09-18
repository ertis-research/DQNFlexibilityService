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
        
        total_actions = 0
        
        for agent in self.agents:
            total_actions += agent.get_action_space()


        self.action_space = spaces.Discrete(total_actions)
        
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
    
    def step(self, action, prediction = 0):
        # The action space is an array that contains the action for each agent
        # Execute one time step within the environment
        self.ac.turnOff()
        self.cs.stop()
        self.sb.stop()
        if self._episode_ended:
            self.reset()
            
        c = 0
        # Execute the action for each agent
        for index in range(len(self.agents)):
            if isinstance(self.agents[index], ChargingStation):
                c+=self.agents[index].step(action[index][0], action[index][1])
            else:
                c+=self.agents[index].step(action[index])
        self.cumulative_consumption += c
        
        self.reward = 0 if self.consumption - prediction == 0 else (0-abs(self.consumption - prediction))/100
        
        return self.getState(), self.reward, self._episode_ended