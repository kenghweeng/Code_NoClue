from collections import defaultdict
import gym
import heapq
import numpy as np
from utils.misc import get_distribution_function
from gym import spaces

class EDPSEnv(gym.Env):
    def __init__(self, config):
        """
        ASSUMPTIONS:
        Treatment order is THE SAME for all patients.
        Each resource type provides only ONE specific treatment. 

        Initialisation parameters for environment:
        acuities               self.data['acuities']
        resources              self.data['resources']
        treatments             self.data['treatments']
        patterns               self.data['patterns']
        patient_distribution   type of distribution modeling patient arrival
        patient_params         parameters for patient distribution
        patient_priority_func  priority function for patient
        warm_up_time           time to exclude from final reward
        max_time               time when simulation cuts off (includes warm-up time)
        seed                   seed of random generator
        """
        super().__init__()

        # Initialisation of parameters
        self.acuities = config["acuities"]
        self.resources = config["resources"]
        self.treatments = config["treatments"]
        self.patterns = config["patterns"]
        self.patient_distribution = config["patient_distribution"]
        self.patient_params = config["patient_params"]
        self.patient_priority_func = config["patient_priority_func"]
        self.warm_up_time = config["warm_up_time"]
        self.max_time = config["max_time"]
        self.set_seed = config["seed"]

        # gym arguments
        self.action_space = spaces.MultiDiscrete(
            [len(self.acuities), len(self.treatments)] for _ in range(len(self.resources))
        )
        empty = np.zeros(self.action_space.nvec)
        high = np.ones(self.action_space.nvec)
        self.observation_space = spaces.Box(empty, high, dtype=np.float64)

        self.reset()

    def step(self, action):
        """
        Action is of the form
        [(acuity, treatment), (acuity, treatment), ...]
        for each respective resource
        """
        for resource, (acuity, treatment) in enumerate(action):
            pool = self.pools[resource][(acuity, treatment)]
            if not pool:
                continue
            patient_id = max(pool, key=self.patient_priority_func)
            pool.remove(patient_id)
            patient = self.patients[patient_id]
            event = self._treat(patient)
            if event:
                heapq.heappush(self.events_heap, event)

    def _treat(self, patient):
        treatment_id = patient.order_ids[patient.order_idx]
        resource_id = self.treatments[treatment_id]['resource_id']
        self.resource_id
        # Process patient
        curr_queue.remove(patient_id)
        self.free_resources[resource_type] -= 1
        patient = self.patients[patient_id]
        event = patient.process(self.time)
        if event:
            heapq.heappush(self.events_heap, event)
        reward = 0 - patient.waiting_time
        reward *= self.weighted_wait[patient.acuity]

        done = self._process_events()

        return self._get_state(), reward, done, self.debug()


    def reset(self):
        self.seed(self.set_seed)
        self.time = 0
        self.patients = []
        self.events_heap = []  # will be used as a heapq of (time, patient id, resource to free, resource to queue)
        self.queues = defaultdict(lambda: defaultdict(list))  # dictionary of resource: (acuity, treatment): list of patient numbers
        self.free_resources = {i: resource['quantity'] for i, resource in enumerate(self.resources)}
        get_spawn_interval = lambda: get_distribution_function(self.rng, self.patient_distribution, self.patient_params)

        # Spawn all patients
        spawn_time = 0
        while spawn_time < self.max_time:
            spawn_time += get_spawn_interval()
            acuity = self.rng.choice(np.arange(self.acuities), p=self.acuities_prob)
            pattern = self.rng.choice(np.arange(self.patterns), p=self.patterns_prob)
            patient_id = len(self.patients)
            patient = Patient(patient_id, spawn_time, acuity, pattern["order_ids"])
            self.patients.append(patient)
            res = self._send_to_queue(self.patients[patient_id])
            heapq.heappush(self.events_heap, (spawn_time, patient_num, None, res))

        self._process_events()
        return self._get_state()

    def _send_to_queue(self, patient):
        treatment_id = patient.order_ids[patient.order_idx]
        resource_id = self.treatments[treatment_id]['resource_id']
        priority = self.patient_priority_func(patient)
        event()
        self.queues[resource_id].append()
        order_idx += 1

    def seed(self, seed=None):
        self.rng = np.random.default_rng(seed)

    def debug(self):
        d = {}
        d["queue"] = self.queues
        d["events"] = self.events_heap
        d["time"] = self.time
        d["free_resources"] = self.free_resources
        return d

    def _get_state(self):
        """
        Normalised fraction of patients of each acuity waiting for each resource
        """
        ans = np.zeros([self.types_resources, self.acuities], dtype=np.float64)
        for res_type in range(self.types_resources):
            for acuity in range(self.acuities):
                ans[res_type][acuity] = len(self.queues[res_type][acuity])
            if np.sum(ans[res_type]) != 0:
                ans[res_type] = ans[res_type] / np.sum(ans[res_type])
        return ans

    def _process_events(self):
        # Check if simulation is over
        if self.time >= self.max_time:
            return True
        
        # No more valid actions and no more events
        if len(self.events_heap) == 0:
            return True

        # Check if there are still valid actions at current timestep
        if any()
        for resource in range(len(self.resources)):
            if self.free_resources[resource] > 0:
                for ac in range(self.acuities):
                    if len(self.queues[res_type][ac]) > 0:
                        return False
        
        # No more valid actions at current timestep, skip to next event
        next_event = self.events_heap[0]
        next_time = next_event[0]

        while True:
            while next_event[0] == next_time:
                next_event = heapq.heappop(self.events_heap)
                _, patient_id, freed_res, next_res = next_event
                if freed_res is not None:  # Need to increase freed resource
                    self.free_resources[freed_res] += 1
                if next_res is not None:  # Need to queue patient to next resource
                    acuity = self.patients[patient_id].acuity
                    self.queues[next_res][acuity].append(patient_id)

                if len(self.events_heap) == 0:
                    self.time = next_time
                    valid_action = False
                    for res_type in range(self.types_resources):
                        if self.free_resources[res_type] > 0:
                            for ac in range(self.acuities):
                                if len(self.queues[res_type][ac]) > 0:
                                    valid_action = True
                                    break
                            if valid_action:
                                break
                    return not valid_action
                next_event = self.events_heap[0]
            
            # Check if there are valid actions
            valid_action = False
            for res_type in range(self.types_resources):
                if self.free_resources[res_type] > 0:
                    for ac in range(self.acuities):
                        if len(self.queues[res_type][ac]) > 0:
                            valid_action = True
                            break
                    if valid_action:
                        break
            if valid_action:
                break
            else:
                next_time = next_event[0]

        # Move to correct time
        self.time = next_time
        return False

class Patient:
    def __init__(self, arrival, acuity, order, treatments, id):
        self.arrival = arrival
        self.start_wait = arrival
        self.acuity = acuity
        self.order = order
        self.id = id
        self.waiting_time = 0

    def get_next_resource(self):
        if self.order:
            return self.order[0]
        raise AttributeError("Patient is no longer queueing!")

    def get_treatment_time(self, resource_type):
        return self.treatments[resource_type]
    
    def process(self, time):
        if self.order:
            waited = time - self.start_wait
            self.waiting_time += waited
            self.start_wait = time + self.treatments[self.order[0]]
            original = self.order.pop(0)
            if len(self.order) > 0:
                return (self.start_wait, self.id, original, self.order[0])
            else:
                return (self.start_wait, self.id, original, None)

