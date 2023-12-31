class ChargingStation():
    def __init__(self, chargingStationID, power = 30.0):
        """
            Initilization method
            : param
                power - Power of the Charge Station
            A station has 2 sockets
        """
        self.chargingStationID = chargingStationID
        self.s1 = 0 # 0 - Available, 1 - Busy, 2 - Charging, 3 - Discharging
        self.s2 = 0 # 0 - Available, 1 - Busy, 2 - Charging, 3 - Discharging
        self.power = power # Default 30 Kw
        self._action_space = 5
        
    def carArrival(self, station):
        """
            Simulation of the arrival of a car at a charging station
            : param
                station - Station where the car is plugged 
        """
        if station == 1 and self.s1 == 0:
            self.s1 = 1
        elif station == 2 and self.s2 == 0:
            self.s2 = 1
        else:
            print("Station is busy")
    def carDeparture(self, station):
        """
            Simulation of the departure of a car at a charging station
            : param
                station - Station where the car is plugged 
        """
        if station == 1 and self.s1 != 0:
            self.s1 = 0
        elif station == 2 and self.s2 != 0:
            self.s2 = 0
        else:
            print("Station was Available yet")
            
    def charge(self, station):
        """
            The car starts charging from the station
            : param
                station - Station where the car is charging 
        """
        if station == 1 and self.s1 != 0:
            self.s1 = 2
        elif station == 2 and self.s2 != 0:
            self.s2 = 2
        return self.consumption()
    def discharge(self, station):
        """
            The car starts discharging from the station
            : param
                station - Station where the car is discharging 
        """
        if station == 1 and self.s1 != 0:
            self.s1 = 3
        elif station == 2 and self.s2 != 0:
            self.s2 = 3
        return self.consumption()  
    def stop(self):
        """
            Stop charging.
        """
        
        self.s1 = 1
        self.s2 = 1
      
    def consumption(self):
        """
            Calculates the consumption of the charging station based on its occupancy and condition.
            : param
                station - Station where the car is charging 
            : return the consumption of the charging station
        """
        if self.s1 == 2 and self.s2 < 3:
            return -self.power
        elif self.s2 == 2 and self.s1 < 3:
            return -self.power
        if self.s1 == 3 and self.s2 == 3:
            return self.power
        if self.s1 == 3 and self.s2 != 2 or self.s1 != 2 and self.s2 == 3:
            return self.power
        else: 
            return 0.0
     
    def step(self, action, station):
        match action:
            case 0:
                return self.carArrival(station)
            case 1:
                return self.carDeparture(station)
            case 2:
                return self.charge(station)
            case _:
                return self.discharge(station)
    def render(self, mode ="human"):
        """
            Print the ev charging station status
        """
        print ("EV Station {} -> Station 1: {}, Station 2 {}, Consumption: {}".format(self.chargingStationID, self.s1, self.s2, self.consumption()))
    def reset(self):
        self.s1 = 0 
        self.s2 = 0
        
    def getChargingStationID(self):
        return self.chargingStationID
    def get_action_space(self):
        return self._action_space
    def getState(self):
        return self.s1, self.s2