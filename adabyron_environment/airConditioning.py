class Airconditioning():
    def __init__(self, airconditioningID):
        self.airconditioningID = airconditioningID
        self.airconditioningState = 0
        self._action_space = 4

    def getAirconditioningID(self):
        return self.airconditioningID
    def get_action_space(self):
        return self._action_space
    def getState(self):
        return self.airconditioningState

    def stop(self):
        self.airconditioningState = 0
        return self.consumption()
    def smallChange(self):
        self.airconditioningState = 1
        return self.consumption()
    def bigChange(self):
        self.ariconditioningState = 2
        return self.consumption()
    def hugeChange(self):
        self.airconditioningState = 3
        return self.consumption()
    
    
    def step(self, action):
        match action:
            case 0:
                return self.stop()
            case 1:
                return self.smallChange()
            case 2:
                return self.bigChange()
            case _:
                return self.hugeChange()  
    def render(self, mode = "human"):
        print("Air conditioning {} -> State {}, Consumption {}".format(self.airconditioningID, self.airconditioningState, self.consumption()))
    def consumption(self):
        if self.airconditioningState == 0:
            return -20
        else:
            #Base consumption + consumption addded per grade
            # 1,13 - It is the coefficient of passage from frigories to watts.
            return -round(20 + self.airconditioningState*15, 2) # Using 15 because it is a quarter-hourly time slot.
    def reset(self):
        self.airconditioningState = 0