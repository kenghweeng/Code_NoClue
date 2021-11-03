import gym
import heapq
import numpy as np
from gym import spaces

class EDPSEnv(gym.Env):
    def __init__(self, config):
        """
        ASSUMPTIONS:
        Treatment order is THE SAME for all patients.
        Each resource type provides only ONE specific treatment. 

        Initialisation parameters for environment:
        resources           dict of resource types (nurse (0)/doctor (1)) to number of resources
        acuities            number of acuities/priority classes
        prob_acuities       list of probabilities for each acuity from 0 to N-1
        weighted_wait       dict of acuities to weight for waiting time
        order               sequence of resource visiting order (i.e. nurse -> doctor is 0 -> 1)
        spawn               RandDist of patients entering ED
        treatment_times     dict of resource types to acuities to RandDist
        max_time            time when simulation cuts off
        set_seed            set seed of random generator

        RandDist is a tuple of Distribution (Poisson, Geometric, etc), then associated params for the distribution
        !!!!! But to simplify I will only use Poisson here !!!!!
        """
        super(EDPSEnv, self).__init__()

        # Initialisation of parameters
        self.resources = config["resources"]
        self.types_resources = len(self.resources)
        self.acuities = config["acuities"]
        self.prob_acuities = config["prob_acuities"]
        assert sum(self.prob_acuities) == 1, "Acuities distribution does not sum to 1"
        self.weighted_wait = config["weighted_wait"]
        self.order = config["order"]
        self.spawn = config["spawn"]
        self.treatment_times = config["treatment_times"]
        self.max_time = config["max_time"]
        self.set_seed = config["set_seed"]

        # gym arguments
        self.action_space = spaces.MultiDiscrete([self.types_resources, self.acuities])
        empty = np.zeros([self.types_resources, self.acuities])
        high = np.ones_like(empty)
        self.observation_space = spaces.Box(empty, high, dtype=np.float64)
        self.seed(self.set_seed)
        
    def step(self, action):
        """
        Action is of the form (resource_type, acuity)
        Assign patient with longest waiting time of the given acuity and waiting for resource_type to resource_type
        """
        resource_type = action[0]
        acuity = action[1]
        assert self.free_resources[resource_type] > 0, "Invalid resource type"
        assert len(self.queues[resource_type][acuity]) > 0, "Invalid patient acuity"
        curr_queue = self.queues[resource_type][acuity]

        # Get longest waiting patient
        patient_queue = list(map(lambda x: self.patients[x], curr_queue))
        waiting = list(map(lambda x: (self.time - x.start_wait + x.waiting_time, x.id), patient_queue))
        patient_id = sorted(waiting, reverse=True)[0][1]

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
        self.patients = {}
        self.events_heap = []  # will be used as a heapq of (time, patient id, free resource, next resource to queue)
        self.queues = {}  # dictionary of resource to acuity to list of patient numbers
        self.free_resources = self.resources.copy()
        for i in range(self.types_resources):
            d = {}
            for j in range(self.acuities):
                d[j] = []
            self.queues[i] = d
        self.time = 0
        self.next_patient_no = 0

        # Spawn all patients
        spawn_time = 0
        while spawn_time < self.max_time:
            patient_num = self._spawn_patient(0)
            start_res = self.patients[patient_num].get_next_resource()
            heapq.heappush(self.events_heap, (spawn_time, patient_num, None, start_res))
            spawn_time += self.rng.poisson(self.spawn)

        self._process_events()
        return self._get_state()

    def seed(self, seed=None):
        self.rng = np.random.default_rng(seed)

    def debug(self):
        d = {}
        d["queue"] = self.queues
        d["events"] = self.events_heap
        d["time"] = self.time
        d["free_resources"] = self.free_resources
        return d

    def _spawn_patient(self, time):
        acuity = self.rng.choice(np.arange(self.acuities), p=self.prob_acuities)
        processing_time = {}
        for res_type in range(self.types_resources):
            processing_time[res_type] = self.rng.poisson(self.treatment_times[res_type][acuity])
        self.patients[self.next_patient_no] = Patient(time, acuity, self.order.copy(), processing_time, self.next_patient_no)
        self.next_patient_no += 1
        return self.next_patient_no - 1

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

        # Check if there are still valid actions at current timestep
        for res_type in range(self.types_resources):
            if self.free_resources[res_type] > 0:
                for ac in range(self.acuities):
                    if len(self.queues[res_type][ac]) > 0:
                        return False
        
        # No more valid actions and no more events
        if len(self.events_heap) == 0:
            return True
        
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
        self.treatments = treatments
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

