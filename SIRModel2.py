import random
#import repast.simphony.engine.environment as env
#import repast.simphony.random as random
from repast4py import random
#import repast.simphony.parameter as params
from repast4py.parameters import params
#import repast.simphony.space.continuous as space
from repast4py import space 
#import repast.simphony.space.grid as grid
#import repast.simphony.space.graph as graph
#import repast.simphony.measure as measure
#from repast4py import measure
#import repast.simphony.distributed.util as dist_utils
#from repast4py import dist_utils

# Define the model parameters
class SIRModel(object):
    def __init__(self, num_agents, initial_infected, transmission_rate, recovery_rate):
        self.num_agents = num_agents
        self.initial_infected = initial_infected
        self.transmission_rate = transmission_rate
        self.recovery_rate = recovery_rate
        self.agents = []
        self.schedule = None
        self.grid = None
        self.graph = None

    # Initialize the agents and their states
    def setup(self):
        self.schedule = random.RandomSchedule()
        self.grid = space.ContinuousSpace(
            "world",
            100,
            100,
            toroidal=True
        )
        self.graph = graph.Network("contact_network")

        # Create agents and add to the schedule, grid, and graph
        for i in range(self.num_agents):
            agent = SIRAgent(i, self)
            self.schedule.add(agent)
            self.grid.moveTo(agent, random.uniform(0, 100), random.uniform(0, 100))
            self.graph.addNode(agent)

        # Set some agents as initially infected
        infected_agents = random.sample(self.agents, self.initial_infected)
        for agent in infected_agents:
            agent.state = "I"

        # Create random contacts between agents
        for agent in self.agents:
            contacts = random.sample(self.agents, random.randint(1, 10))
            for contact in contacts:
                self.graph.addEdge(agent, contact)

    # Define the model step
    def step(self):
        # Iterate over all agents and update their states
        for agent in self.agents:
            if agent.state == "S":
                # Check if the agent is infected by a neighboring infected agent
                for neighbor in self.graph.getNeighbors(agent):
                    if neighbor.state == "I" and random.random() < self.transmission_rate:
                        agent.state = "I"
                        break
            elif agent.state == "I":
                # Check if the agent recovers
                if random.random() < self.recovery_rate:
                    agent.state = "R"
        # Advance the schedule to the next step
        self.schedule.step()

# Define the agent behavior
class SIRAgent(object):
    def __init__(self, unique_id, model):
        self.unique_id = unique_id
        self.model = model
        self.state = "S"

# Define the experiment
def run_experiment():
    # Set up the model parameters
    num_agents = params.createIntegerParameter("num_agents", 100)
    initial_infected = params.createIntegerParameter("initial_infected", 1)
    transmission_rate = params.createDoubleParameter("transmission_rate", 0.1)
    recovery_rate = params.createDoubleParameter("recovery_rate", 0.05)

    # Define the experiment
    experiment = measure.MeasureGroup("SIR Model")
    experiment.addMeasure(measure.IterationCountMeasure())
    experiment.addMeasure(measure.PopulationMeasure())
    experiment.addMeasure(measure.SpaceMeasure("world"))
    experiment.addMeasure(measure.NetworkMeasure("contact_network"))

    # Create the model
    model = SIRModel(
        num_agents.getValue(),
        initial_infected.getValue(),
        transmission_rate.getValue(),
        recovery_rate.getValue()
)
    
SIRModel.setup()

def __init__(self, comm, params):
        self.runner = schedule.init_schedule_runner(comm)
        self.runner.schedule_repeating_event(1, 1, self.step)
        self.runner.schedule_stop(params['stop.at'])
        self.runner.schedule_end_event(self.at_end)

# Create the runner
self.runner = dist_utils.DistributedRunner()
self.runner.setBatchSize(1)
self.runner.setNumThreads(1)

# Run the experiment
self.runner.runInitialization()
self.runner.run(SIRModel, experiment, 500)
self.runner.runFinalization()

run_experiment()



