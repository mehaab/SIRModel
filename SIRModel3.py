import random
import numpy as np
import repast.simphony.engine.schedule as schedule
import repast.simphony.parameter as params
import repast.simphony.random as rand
import repast.simphony.space.grid as grid
import repast.simphony.space.continuous as cont
import repast.simphony.visualization.gis.GisUtilities as gis
import repast.simphony.visualization.gis.AOI as aoi

from repast.simphony.engine import DistributedObject
from repast.simphony.data_loader import ExcelReader
from repast.simphony.space import WrapAroundBorders

class SIRModel(DistributedObject):
    def __init__(self):
        super().__init__()
        self.grid = None
        self.susceptible = None
        self.infected = None
        self.recovered = None
        self.num_agents = None
        self.schedule = schedule.RandomActivation(self.context)
        self.beta = 0.5
        self.gamma = 0.1
        self.rumor_threshold = 0.5

    def setup(self):
        self.grid = grid.Grid2D(100, 100, toroidal=True, grid_style=WrapAroundBorders)
        self.susceptible = self.grid.add_agents(200, SusceptibleAgent)
        self.infected = []
        self.recovered = []
        self.num_agents = 200
        self.rumor_start_agents = 20
        self.init_rumors()
        self.init_infected()
        self.init_schedules()

    def init_rumors(self):
        excel_reader = ExcelReader("rumor-data.xlsx")
        rumors = excel_reader.read_sheet("rumors")
        self.rumors = {}
        for rumor in rumors:
            self.rumors[rumor["id"]] = Rumor(rumor["text"], rumor["truthiness"])

    def init_infected(self):
        for i in range(self.num_agents):
            if i < self.rumor_start_agents:
                self.susceptible[i].rumor = self.rumors[i]
            else:
                self.infected.append(self.susceptible[i])
        rand.shuffle(self.infected)
        for i in range(10):
            self.infect(self.infected[i])

    def init_schedules(self):
        self.schedule.add(self.update_rumors)
        self.schedule.add(self.update_sir)

    def update_rumors(self):
        for a in self.susceptible:
            if a.rumor:
                neighbors = self.grid.get_neighbors(a.position, moore=True, include_center=True)
                if any([n.infected for n in neighbors if isinstance(n, InfectedAgent)]):
                    if random.random() < self.rumor_threshold:
                        a.rumor.truthiness *= 1.1
                    else:
                        a.rumor.truthiness *= 0.9

    def update_sir(self):
        for a in self.infected:
            if random.random() < self.gamma:
                a.infected = False
                a.recovered = True
                self.infected.remove(a)
                self.recovered.append(a)
            else:
                neighbors = self.grid.get_neighbors(a.position, moore=True, include_center=False)
                for n in neighbors:
                    if isinstance(n, SusceptibleAgent) and random.random() < self.beta:
                        self.infect(n)

    def infect(self, a):
        a.infected = True
        self.infected.append(a)
        self.susceptible.remove(a)

class Agent(DistributedObject):
    def __init__(self, unique_id, model):
        super().__
from repast.simphony.random import RandomHelper
from repast.simphony.space.grid import GridPoint
from repast.simphony.space.grid import GridBuilder
from repast.simphony.space import WrapAroundBorders
from repast.simphony.engine.schedule import ScheduleParameters
from repast.simphony.engine.schedule import RandomSequence
from repast.simphony.parameter import Parameters
from repast.simphony.data2 import DataWriter
from repast.simphony.visualization import Display

class SIRModel:
    def __init__(self, gridSize, probInfection, probRecovery, probSpreadRumor, numRumors, rumorSpreaderPercent, numSteps, logFileName):
        # Define the parameters
        self.gridSize = gridSize
        self.probInfection = probInfection
        self.probRecovery = probRecovery
        self.probSpreadRumor = probSpreadRumor
        self.numRumors = numRumors
        self.rumorSpreaderPercent = rumorSpreaderPercent
        self.numSteps = numSteps
        self.logFileName = logFileName

        # Initialize the simulation
        self.initialize()

    def initialize(self):
        # Create the grid
        border = WrapAroundBorders(self.gridSize, self.gridSize)
        self.grid = GridBuilder.createGrid2D(border, GridPoint)

        # Add agents to the grid
        self.agents = []
        for x in range(self.gridSize):
            for y in range(self.gridSize):
                agent = SIRAgent(x, y, self.probInfection, self.probRecovery, self.probSpreadRumor, self.numRumors, self.rumorSpreaderPercent)
                self.grid.moveTo(agent, x, y)
                self.agents.append(agent)

        # Create the schedule
        self.schedule = RandomSequence()
        self.scheduleParams = ScheduleParameters()
        self.scheduleParams.setRandom(RandomHelper.getUniform())
        self.scheduleParams.setOrderMode(ScheduleParameters.ORDER_MODE.RANDOM)

        # Add the agents to the schedule
        for agent in self.agents:
            self.schedule.add(agent, agent.step, self.scheduleParams)

        # Initialize the log file
        self.logFile = open(self.logFileName, 'w')
        self.logFile.write("Step, NumSusceptible, NumInfected, NumRecovered, NumRumorsSpread\n")

        # Initialize the visualization
        self.display = Display("SIR Model", self)
        self.display.setBackgroundColor(255, 255, 255)

    def run(self):
        # Run the simulation
        for i in range(self.numSteps):
            self.schedule.step()
            self.log(i)
            self.display.update()

        # Close the log file
        self.logFile.close()

    def log(self, step):
        # Log the current state of the simulation
        numSusceptible = sum(1 for agent in self.agents if agent.state == SIRAgent.State.SUSCEPTIBLE)
        numInfected = sum(1 for agent in self.agents if agent.state == SIRAgent.State.INFECTED)
        numRecovered = sum(1 for agent in self.agents if agent.state == SIRAgent.State.RECOVERED)
        numRumorsSpread = sum(1 for agent in self.agents if agent.rumorsSpread > 0)
        self.logFile.write(f"{step}, {numSusceptible}, {numInfected}, {numRecovered}, {numRumorsSpread}\n")

class SIRAgent:
    class State:
        SUSCEPTIBLE = 0
        INFECTED = 1
        RECOVERED = 2

import repast.simphony.engine.schedule as sched

# Define the agents
class Person(repast.simphony.data_containers.AgentId):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.infected = False
        self.recovered = False
        self.immune = False
        self.days_infected = 0
    
    def infect(self):
        self.infected = True
    
    def recover(self):
        self.infected = False
        self.recovered = True
        
    def step(self):
        if self.infected:
            self.days_infected += 1
            if self.days_infected >= 14:
                self.recover()
                
    def update(self):
        neighbors = self.model.grid.get_neighbors_within_distance(self, 1)
        for neighbor in neighbors:
            if neighbor.infected and not self.immune:
                if np.random.rand() < model.transmission_prob:
                    self.infect()
                    break
        if self.infected:
            self.step()
            self.model.infected_count += 1
        elif self.recovered:
            self.immune = True
            
class SIRModel(repast.simphony.data_containers.Model):
    def __init__(self, height, width, density, transmission_prob, recovery_prob, initial_infected_count):
        super().__init__()
        self.height = height
        self.width = width
        self.density = density
        self.transmission_prob = transmission_prob
        self.recovery_prob = recovery_prob
        self.initial_infected_count = initial_infected_count
        self.infected_count = 0
        self.schedule = sched.RandomActivation(self)
        self.grid = repast.simphony.space.grid.Grid2D(self.width, self.height, torus=True)
        self.data_collector = repast.simphony.data_collectors.DataCollector()
        
    def setup(self):
        for i in range(self.width):
            for j in range(self.height):
                if np.random.rand() < self.density:
                    p = Person(self.next_id(), self)
                    self.grid.place_agent(p, (i,j))
                    self.schedule.add(p)
                    if self.initial_infected_count > 0:
                        p.infect()
                        self.initial_infected_count -= 1
        self.running = True
        self.data_collector.collect(self)
        
    def step(self):
        self.infected_count = 0
        self.schedule.step()
        if self.infected_count == 0:
            self.running = False
            
    def run(self, steps=1000):
        for i in range(steps):
            if self.running:
                self.step()
            else:
                break
        self.data_collector.collect(self)

if __name__ == '__main__':
    model = SIRModel(50, 50, 0.6, 0.5, 0.1, 5)
    model.setup()
    model.run()
