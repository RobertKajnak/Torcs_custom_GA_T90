from pytocl.driver import Driver
#from pytocl.car import State, Command, MPS_PER_KMH
#import logging
#from sklearn.externals import joblib
import mlp_evo as mlp
from pytocl.driver import Driver
from pytocl.car import State, Command, MPS_PER_KMH
from stopwatch import Stopwatch
import csv

cap = lambda x: x if x>=0 else 0 

class MyDriver(Driver):
    # Override the `drive` method to create your own driver
    my_esn = None;
    regr=None;
    
    def __init__(self):
        sw = Stopwatch('init');
           
        #self.net = mlp.MLP(6,20,3)           
        #self.net.save_to_file("thenet.net")     
        self.net = mlp.load_MLP_from_file("thenet.net")  
        sw.clock()
        
    def drive(self,carstate: State) -> Command:
        command = Command()
        #print(carstate.distances_from_edge)        
        
        input = [carstate.speed_x, carstate.distance_from_center, carstate.angle]\
        + [carstate.distances_from_edge[1]] + [carstate.distances_from_edge[5]] +\
        [carstate.distances_from_edge[9]] + [carstate.distances_from_edge[13]] + \
       [ carstate.distances_from_edge[17]]
        
        output = self.net.ffwd(input)
        accelerate = cap (1 - output[0] * 2)
        brake = cap(output[0] * 2 - 1)
        steering = output[1]-output[2]
        
        if carstate.current_lap_time<6:
            accelerate=1
            brake=0
            
        #if carstate.speed_x<=1 and brake>=0.1:
        #    carstate.gear=-1

        print(accelerate, brake, steering)
        command.accelerator = accelerate
        command.brake = brake
        command.steering = steering
        
        #command.accelerator = 1
        if carstate.rpm > 8000:
            command.gear = carstate.gear + 1

        if carstate.rpm < 2500 and carstate.gear > 0:
            command.gear = carstate.gear - 1

        if not command.gear:
            command.gear = carstate.gear or 1
        
        with open('fitness_parameter.csv', 'w') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            writer.writerow([carstate.distance_raced, carstate.current_lap_time, carstate.race_position])
        return command
        
        

## 