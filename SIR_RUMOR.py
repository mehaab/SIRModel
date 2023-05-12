import networkx as nx
from typing import Dict
from mpi4py import MPI
import numpy as np
from dataclasses import dataclass

from repast4py.network import write_network, read_network
from repast4py import context as ctx
from repast4py import core, random, schedule, logging, parameters


def generate_network_file(fname: str, n_ranks: int, n_agents: int):
    """Generates a network file using repast4py.network.write_network.

    Args:
        fname: the name of the file to write to
        n_ranks: the number of process ranks to distribute the file over
        n_agents: the number of agents (node) in the network
    """
    g = nx.connected_watts_strogatz_graph(n_agents, 2, 0.25)
    try:
        import nxmetis
        write_network(g, 'rumor_network', fname, n_ranks, partition_method='metis')
    except ImportError:
        write_network(g, 'rumor_network', fname, n_ranks)


model = None
class SIRAgent(core.Agent):
    def __init__(self, nid: int, agent_type: int, rank: int, state: str):
        super().__init__(nid, agent_type, rank)
        self.state = state
        self.next_state = None

    def save(self):
        """Saves the state of this agent as tuple.

        A non-ghost agent will save its state using this
        method, and any ghost agents of this agent will
        be updated with that data (self.state).
        Returns:
            The agent's state
        """
        return (self.uid, self.state)

    def update(self, data: str):
        """Updates the state of this agent when it is a ghost
        agent on some rank other than its local one.

        Args:
            data: the new agent state
        """
        self.state = data

def create_sir_agent(nid, agent_type, rank, **kwargs):
    return SIRAgent(nid, agent_type, rank)


def restore_agent(agent_data):
    uid = agent_data[0]
    return SIRAgent(uid[0], uid[1], uid[2], agent_data[1])

@dataclass
class SirCounts:
    total_rumor_spreaders: int
    new_rumor_spreaders: int

class Model:

    def __init__(self, comm, params):
        self.runner = schedule.init_schedule_runner(comm)
        self.runner.schedule_repeating_event(1, 1, self.step)
        self.runner.schedule_stop(params['stop.at'])
        self.runner.schedule_end_event(self.at_end)

        fpath = params['network_file']
        self.context = ctx.SharedContext(comm)
        read_network(fpath, self.context, create_sir_agent, restore_agent)
        self.net = self.context.get_projection('sir_network')

        self.susceptible_agents = []
        self.infected_agents = []
        self.removed_agents = []
        self.rank = comm.Get_rank()

        self._seed_infection(params['initial_infection_count'], comm)

        self.counts = SirCounts(0, len(self.infected_agents), 0)
        loggers = logging.create_loggers(self.counts, op=MPI.SUM, rank=self.rank)
        self.data_set = logging.ReducingDataSet(loggers, comm, params['counts_file'])
        self.data_set.log(0)

        self.infection_prob = params['infection_probability']
        self.recovery_prob = params['recovery_probability']

    def _seed_infection(self, init_infection_count: int, comm):
        world_size = comm.Get_size()
        # np array of world size, the value of i'th element of the array
        # is the number of infections to seed on rank i.
        infection_counts = np.zeros(world_size, np.int32)
        if (self.rank == 0):
            for _ in range(init_infection_count):
                idx = random.default_rng().integers(0, high=world_size)
                infection_counts[idx] += 1

        infection_count = np.empty(1, dtype=np.int32)
        comm.Scatter(infection_counts, infection_count, root=0)

        for agent in self.context.agents(shuffle=True):
            if infection_count[0] > 0:
                agent.state = 'I'
                self.infected_agents.append(agent)
                infection_count[0] -= 1
            else:
                agent.state = 'S'
                self.susceptible_agents.append(agent)

    def at_end(self):
        self.data_set.close()

    def step(self):
        self._infect()
        self._recover()
        self.counts.total += 1
        self.counts.infected = len(self.infected_agents)
        self.counts.removed = len(self.removed_agents)
        self.data_set.log(self.runner.schedule.tick)
        self.context.synchronize(restore_agent)

    def _infect(self):
        rng = random.default_rng()
        for neighbor in self.neighbors:
            if not neighbor.is_infected and rng.random() < self.transmission_rate:
                neighbor.is_infected = True
                neighbor.infection_timer = self.infection_duration

def run(params: Dict):
    global model
    model = Model(MPI.COMM_WORLD, params)
    model.start()

if __name__ == "__main__":
    parser = parameters.create_args_parser()
    args = parser.parse_args()
    params = parameters.init_params(args.parameters_file, args.parameters)
    run(params)
