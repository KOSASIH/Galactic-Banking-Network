import numpy as np
from skfuzzy import control as ctrl
from skfuzzy import fuzzyvariable as fuzz

class GalacticFuzzyLogic:
    def __init__(self):
        self.distance = ctrl.Antecedent(np.arange(0, 101, 1), 'distance')
        self.velocity = ctrl.Antecedent(np.arange(0, 201, 1), 'velocity')
        self.galaxy_type = ctrl.Consequent(np.arange(0, 3, 1), 'galaxy_type')

        self.distance.automf(3)
        self.velocity.automf(3)
        self.galaxy_type.automf(3)

        self.rule1 = ctrl.Rule(self.distance['poor'] & self.velocity['poor'], self.galaxy_type['Spiral'])
        self.rule2 = ctrl.Rule(self.distance['avg'] & self.velocity['avg'], self.galaxy_type['Elliptical'])
        self.rule3 = ctrl.Rule(self.distance['good'] & self.velocity['good'], self.galaxy_type['Irregular'])

        self.galaxy_type_ctrl = ctrl.ControlSystem([self.rule1, self.rule2, self.rule3])
        self.galaxy_type_sim = ctrl.ControlSystemSimulation(self.galaxy_type_ctrl)

    def evaluate(self, distance, velocity):
        self.galaxy_type_sim.input['distance'] = distance
        self.galaxy_type_sim.input['velocity'] = velocity
        self.galaxy_type_sim.compute()
        return self.galaxy_type_sim.output['galaxy_type']

def main():
    gfl = GalacticFuzzyLogic()

    distance = 50
    velocity = 100
    galaxy_type = gfl.evaluate(distance, velocity)
    print("Galaxy Type:", galaxy_type)

    distance = 80
    velocity = 150
    galaxy_type = gfl.evaluate(distance, velocity)
    print("Galaxy Type:", galaxy_type)

    distance = 20
    velocity = 50
    galaxy_type = gfl.evaluate(distance, velocity)
    print("Galaxy Type:", galaxy_type)

if __name__ == "__main__":
    main()
