from __future__ import print_function

import collections

# Import Python wrapper for or-tools CP-SAT solver.
import time

# import wandb
from ortools.sat.python import cp_model
from ortools.sat.python.cp_model import CpSolverSolutionCallback


class WandbCallback(CpSolverSolutionCallback):
    """Display the objective value and time of intermediate solutions."""

    def __init__(self):
        CpSolverSolutionCallback.__init__(self)
        self.__solution_count = 0
        self.__start_time = time.time()
        # wandb.log({'time': 0, 'solution_count': self.__solution_count})

    def on_solution_callback(self):
        """Called on each new solution."""
        current_time = time.time()
        obj = self.ObjectiveValue()
        self.__solution_count += 1
        # wandb.log({'time': current_time - self.__start_time, 'make_span': obj, 'solution_count': self.__solution_count})

    def solution_count(self):
        """Returns the number of solutions found."""
        return self.__solution_count

import sys
try: instance_filename = sys.argv[1]
except: instance_filename = 'covid01'

machine2label=['registration', 'x_ray', 'consultation', 'ct_scan', 'dispensary']

def MinimalJobshopSat(instance_config= {'instance_path': f'JSS/instances/{instance_filename}'}):
    # wandb.init(config=instance_config)
    # config = wandb.config
    config = instance_config
    """Minimal jobshop problem."""
    # Create the model.
    model = cp_model.CpModel()

    jobs_data = []
    machines_count = 0

    instance_file = open(config['instance_path'], 'r')
    line_str = instance_file.readline()
    line_cnt = 1
    while line_str:
        data = []
        split_data = line_str.split()
        if line_cnt == 1:
            jobs_count, machines_count = int(split_data[0]), int(split_data[1])
        else:
            i = 0
            while i < len(split_data):
                machine, time = int(split_data[i]), int(split_data[i + 1])
                data.append((machine, time))
                i += 2
            jobs_data.append(data)
        line_str = instance_file.readline()
        line_cnt += 1
    instance_file.close()

    all_machines = range(machines_count)

    # Computes horizon dynamically as the sum of all durations.
    horizon = sum(task[1] for job in jobs_data for task in job)

    # Named tuple to store information about created variables.
    task_type = collections.namedtuple('task_type', 'start end interval')
    # Named tuple to manipulate solution information.
    assigned_task_type = collections.namedtuple('assigned_task_type',
                                                'start job index duration')

    # Creates job intervals and add to the corresponding machine lists.
    all_tasks = {}
    machine_to_intervals = collections.defaultdict(list)

    for job_id, job in enumerate(jobs_data):
        for task_id, task in enumerate(job):
            machine = task[0]
            duration = task[1]
            suffix = '_%i_%i' % (job_id, task_id)
            start_var = model.NewIntVar(0, horizon, 'start' + suffix)
            end_var = model.NewIntVar(0, horizon, 'end' + suffix)
            interval_var = model.NewIntervalVar(start_var, duration, end_var,
                                                'interval' + suffix)
            all_tasks[job_id, task_id] = task_type(
                start=start_var, end=end_var, interval=interval_var)
            machine_to_intervals[machine].append(interval_var)

    # Create and add disjunctive constraints.
    for machine in all_machines:
        model.AddNoOverlap(machine_to_intervals[machine])

    # Precedences inside a job.
    for job_id, job in enumerate(jobs_data):
        for task_id in range(len(job) - 1):
            model.Add(all_tasks[job_id, task_id +
                                1].start >= all_tasks[job_id, task_id].end)

    # Makespan objective.
    obj_var = model.NewIntVar(0, horizon, 'makespan')
    model.AddMaxEquality(obj_var, [
        all_tasks[job_id, len(job) - 1].end
        for job_id, job in enumerate(jobs_data)
    ])
    model.Minimize(obj_var)

    # Solve model.
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 600.0
    # wandb_callback = WandbCallback()
    status = solver.SolveWithSolutionCallback(model, None)
    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        # Create one list of assigned tasks per machine.
        assigned_jobs = collections.defaultdict(list)
        for job_id, job in enumerate(jobs_data):
            for task_id, task in enumerate(job):
                machine = task[0]
                assigned_jobs[machine].append(
                    assigned_task_type(start=solver.Value(
                        all_tasks[job_id, task_id].start),
                                    job=job_id,
                                    index=task_id,
                                    duration=task[1]))

        solution_sequence = []
        for machine in all_machines:
            assigned_jobs[machine].sort()
            solution_sequence.append([assigned_task.job for assigned_task in assigned_jobs[machine]])
        # print(solution_sequence)
    else:
        print('No solution found.')
    
    import gym
    import imageio

    # http://optimizizer.com/solution.php?name=ta01&UB=1231&problemclass=ta
    env = gym.make('JSSEnv:jss-v1', env_config={'instance_path': f'JSS/instances/{instance_filename}'})
    env.reset()
    # for every machine give the jobs to process in order for every machine
    done = False
    job_nb = len(solution_sequence[0])
    machine_nb = len(solution_sequence)
    index_machine = [0 for _ in range(machine_nb)]
    step_nb = 0
    images = []
    while not done:
        # if we haven't performed any action, we go to the next time step
        no_op = True
        for machine in range(len(solution_sequence)):
            if done:
                break
            if env.machine_legal[machine] and index_machine[machine] < job_nb:
                action_to_do = solution_sequence[machine][index_machine[machine]]
                if env.needed_machine_jobs[action_to_do] == machine and env.legal_actions[action_to_do]:
                    no_op = False
                    state, reward, done, _ = env.step(action_to_do)
                    index_machine[machine] += 1
                    step_nb += 1
                    # temp_image = env.render(machine2label=machine2label).to_image()
                    # images.append(imageio.imread(temp_image))
        if no_op and not done:
            previous_time_step = env.current_time_step
            env.increase_time_step()
    make_span = env.last_time_step
    # print("Completed simulation")
    # imageio.mimsave(f"JSS/gifs/{instance_filename}_or.gif", images, format='GIF', fps=2)
    print(make_span)
    env.reset()



if __name__ == "__main__":
    MinimalJobshopSat()