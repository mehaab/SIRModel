from typing import Dict, Tuple
from mpi4py import MPI
import numpy as np
from dataclasses import dataclass
import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt

from repast4py import core, random, space, schedule, logging, parameters
from repast4py import context as ctx
import repast4py
from repast4py.space import DiscretePoint as dpt


@dataclass
class MeetLog:
    total_meets: int = 0
    min_meets: int = 0
    max_meets: int = 0


class Person(core.Agent):
    TYPE = 0
    OFFSETS = np.array([-1, 1])

    def __init__(self, local_id: int, rank: int, pt: dpt, model):
        super().__init__(id=local_id, type=Person.TYPE, rank=rank)
        self.pt = pt
        self.infected = False
        self.recovered = False
        self.model = model

    def save(self) -> Tuple:
        """Saves the state of this Person as a Tuple.

        Returns:
            The saved state of this Person.
        """
        return (self.uid, self.infected, self.recovered, self.pt.coordinates)

    def walk(self, grid):
        # choose two elements from the OFFSET array
        # to select the direction to walk in the
        # x and y dimensions
        xy_dirs = random.default_rng().choice(Person.OFFSETS, size=2)
        self.pt = grid.move(self, dpt(self.pt.x + xy_dirs[0], self.pt.y + xy_dirs[1], 0))

    def infect(self):
        self.infected = True
        self.model.infected_count += 1
        self.model.susceptible_count -= 1

    def recover(self):
        self.recovered = True
        self.infected = False
        self.model.recovered_count += 1
        self.model.infected_count -= 1

    def step(self):
        if self.recovered:
            return

        if self.infected:
            self.model.total_infected += 1
            if self.model.random_generator.random() < self.model.recovery_rate:
                self.recover()
            else:
                self.walk(self.model.grid)
        else:
            neighbors = self.model.grid.get_neighbors_within_distance(self, self.model.infection_radius)
            for neighbor in neighbors:
                if not neighbor.infected:
                    if self.model.random_generator.random() < self.model.infection_rate:
                        neighbor.infect()
                        self.model.total_meets += 1
                        break
            self.walk(self.model.grid)


person_cache = {}


def restore_person(person_data: Tuple, model):
    """
    Args:
        person_data: tuple containing the data returned by Person.save.
    """
    # uid is a 3 element tuple: 0 is id, 1 is type, 2 is rank
    uid = person_data[0]
    pt_array = person_data[3]
    pt = dpt(pt_array[0], pt_array[1], 0)

    if uid in person_cache:
        person = person_cache[uid]
    else:
        person = Person(uid[0], uid[2], pt, model)
        person_cache[uid] = person

    person.infected = person_data[1]
    person.recovered = person_data[2]
    person.pt = pt
    return person


class SIRModel:
    """
    The SIRModel class encapsulates the simulation, and is
    responsible for initialization (scheduling events, creating agents,
and the grid the agents inhabit), and the overall iterating
behavior of the model.

Args:
    comm: the mpi communicator over which the model is distributed.
    params: the simulation input parameters
"""

def __init__(self, comm: MPI.Intracomm, params: Dict):
    # create the schedule
    self.runner = schedule.init_schedule_runner(comm)
    self.runner.schedule_repeating_event(1, 1, self.step)
    self.runner.schedule_repeating_event(1.1, 10, self.log_agents)
    self.runner.schedule_stop(params['stop.at'])

    # create the space
    self.space = space.Grid2D(
        params['grid.width'],
        params['grid.height'],
        False,
        x_min=params['grid.x_min'],
        y_min=params['grid.y_min']
    )

    # create the context
    self.ctx = ctx.Context()

    # set up logging
    self.logger = logging.getLogger("SIRModel Logger")
    self.logger.setLevel(logging.INFO)

    # set up parameters
    self.params = parameters.Parameters()

    # create agents
    for i in range(params['model.population']):
        pt = self.space.random_cell()
        person = Person(i, comm.rank, pt, self)
        self.ctx.add(person)
        self.space.place_agent(person, pt)

    # infect a random person
    infected_person = random.choice(self.ctx.agents)
    infected_person.infect()

    # set up meet log
    self.meet_log = MeetLog()

    # set up counts
    self.infected_count = 1
    self.recovered_count = 0

def step(self):
    # move all people
    for person in self.ctx.agents:
        person.walk(self.space)

    # have people meet each other
    for person in self.ctx.agents:
        neighbors = self.space.get_neighbors_within_distance(
            person.pt, 1, True)
        for neighbor in neighbors:
            self.meet(person, neighbor)

    # recover people who were infected
    for person in self.ctx.agents:
        if person.infected and random.random() < self.params.get('recovery.rate'):
            person.recover()

def meet(self, person, other):
    self.meet_log.total_meets += 1
    if person.infected and not other.recovered:
        other.infect()
    elif other.infected and not person.recovered:
        person.infect()

def log_agents(self):
    self.logger.info(f"Rank {self.comm.rank} has {len(self.ctx.agents)} agents.")

def plot_results(self):
    # plot the number of infected and recovered people over time
    counts = np.array((self.infected_count, self.recovered_count))
    counts = np.zeros_like(counts)
    counts[0] = self.infected_count
    counts[1] = self.recovered_count

    fig, ax = plt.subplots()
    ax.plot(counts[0], 'r', label='Infected')
    ax.plot(counts[1], 'g', label='Recovered')
    ax.set_xlabel('Time')
    ax.set_ylabel('Number of People')
    ax.legend()
    plt.show()

