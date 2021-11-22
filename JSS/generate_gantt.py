import datetime
import pandas as pd
import gym
import numpy as np
import plotly.figure_factory as ff
import sys
import pickle
import random
import imageio
from env import JssEnv

class GanttChart:
    def __init__(self, instance):
        self.instance = instance
        self.machine2label = ['daily_rounds', 'x_ray', 'consultation', 'ct_scan', 'dispensary']
        self.start_timestamp = datetime.datetime.fromtimestamp(5400).timestamp()
        self.solution = None
        
        try:
            with open(f'solutions/{self.instance}_sol.pkl', 'rb') as f:
                dict = pickle.load(f)
                self.solution, self.solution_makespan = dict['solution'], dict['makespan']

        except:
            print("Solution not generated yet, need to use main.py to train RL agent first")

    def create_job_matrix(self):
        try:
            instance_file = open(f'instances/{self.instance}', 'r')
        except:
            print(f'\nInstance {self.instance} not found, setting default instance to be covid5_1.txt... \n')
            self.instance = 'covid5_1'
            instance_file = open(f'instances/{self.instance}', 'r')

        line_str = instance_file.readline()
        line_cnt = 1
        while line_str:
            split_data = line_str.split()
            if line_cnt == 1:
                self.jobs, self.machines = int(split_data[0]), int(split_data[1])
                # matrix which store tuple of (machine, length of the job)
                self.instance_matrix = np.zeros((self.jobs, self.machines), dtype=(np.int, 2))
            else:
                # couple (machine, time)
                assert len(split_data) % 2 == 0
                # each jobs must pass a number of operation equal to the number of machines
                assert len(split_data) / 2 == self.machines
                i = 0
                # we get the actual jobs
                job_nb = line_cnt - 2
                while i < len(split_data):
                    machine, time = int(split_data[i]), int(split_data[i + 1])
                    self.instance_matrix[job_nb][i // 2] = (machine, time)
                    i += 2
            line_str = instance_file.readline()
            line_cnt += 1
        instance_file.close()

    def generate_gantt_csv(self):
        if self.solution is None:
            return "You need to first run main.py to generate solution for the input instance"

        head_df = []
        df = []
        for job in range(self.jobs):
            # set initial jobs to all appear in gantt
            for i in range(self.machines):
                dict_op = dict()
                dict_op["Task"] = 'Patient {}'.format(job+1)
                dict_op["Task_Num"] = job # used for pivoting later
                dict_op["Start"] = datetime.datetime.fromtimestamp(self.start_timestamp-1e-9)
                dict_op["Finish"] = datetime.datetime.fromtimestamp(self.start_timestamp-1e-9)
                dict_op["Resource"] = self.machine2label[self.instance_matrix[job][i][0]]
                dict_op["Resource_Num"] = self.instance_matrix[job][i][0]
                head_df.append(dict_op)

            i = 0
            while i < self.machines and self.solution[job][i] != -1:
                dict_op = dict()
                dict_op["Task"] = 'Patient {}'.format(job+1)
                dict_op["Task_Num"] = job # used for pivoting later
                start_sec = self.start_timestamp + 60 * self.solution[job][i]
                finish_sec = start_sec + 60 * self.instance_matrix[job][i][1]
                dict_op["Start"] = datetime.datetime.fromtimestamp(start_sec)
                dict_op["Finish"] = datetime.datetime.fromtimestamp(finish_sec)
                dict_op["Resource"] = self.machine2label[self.instance_matrix[job][i][0]]
                dict_op["Resource_Num"] = self.instance_matrix[job][i][0]
                df.append(dict_op)
                i += 1

        if len(df) > 0:
            self.df = pd.DataFrame(df)
            self.plot_df = pd.DataFrame(head_df + df)
            clean_df = self.df.drop(columns=["Task_Num", "Resource_Num"], axis=1)
            clean_df["Start"] = clean_df["Start"].dt.strftime("%H:%M:%S")
            clean_df["Finish"] = clean_df["Finish"].dt.strftime("%H:%M:%S")
            clean_df.to_csv(f'schedules/{self.instance}.csv', index=False)
            print(f"Saved the optimal schedule {self.instance}.csv to schedules folder!")
            # return self.df

    def generate_chart(self):
        self.colors = [
            tuple([random.random() for _ in range(3)]) for _ in range(self.machines)
        ]
            
        fig = ff.create_gantt(self.plot_df, index_col='Resource', colors=self.colors, show_colorbar=True,
                                group_tasks=True)
        fig.update_layout(
            title='Patient scheduling',
            xaxis_tickformat= '%H:%M:%S',
            xaxis_title=f'Time (24-hour format) with a makespan of {self.solution_makespan} minutes',
        )

        fig.update_yaxes(autorange="reversed")  # otherwise tasks are listed from the bottom up
        print(f"Saved {self.instance}.PNG to images folder!")
        imageio.imsave(f'images/{self.instance}.png', imageio.imread(fig.to_image()))


    def generate_gif(self):
        # we convert the timings solution (found in self.solution) to the solutions ordered in sequence, 
        # refer to http://jobshop.jjvh.nl/explanation.php

        sol_seq = []
        # Conversion of timings to solution sequence instead
        sorted = self.df.sort_values(by=['Resource_Num', 'Start'])
        df_grouped = sorted.groupby("Resource_Num")

        for _, group in df_grouped:
            sol_seq.append([int(job) for job in group['Task_Num'].values])

        # print(sol_seq) for debugging

        #################################
        # Code block for GIF generation # 
        #################################

        env = JssEnv(env_config={'instance_path': f"instances/{self.instance}"})
        env.reset()
        
        done = False
        job_nb = len(sol_seq[0])
        machine_nb = len(sol_seq)
        index_machine = [0 for _ in range(machine_nb)]

        step_nb = 0
        images = []
        while not done:
            # if we haven't performed any action, we go to the next time step
            no_op = True
            for machine in range(len(sol_seq)):
                if done:
                    break
                if env.machine_legal[machine] and index_machine[machine] < job_nb:
                    action_to_do = sol_seq[machine][index_machine[machine]]
                    if env.needed_machine_jobs[action_to_do] == machine and env.legal_actions[action_to_do]:
                        no_op = False
                        state, reward, done, _ = env.step(action_to_do)
                        index_machine[machine] += 1
                        step_nb += 1
                        temp_image = env.render(machine2label=self.machine2label).to_image()
                        images.append(imageio.imread(temp_image))
            if no_op and not done:
                previous_time_step = env.current_time_step
                env.increase_time_step()

        imageio.mimsave(f"images/{self.instance}.gif", images, format='GIF', fps=2)
        print(f"Saved {self.instance}.GIF to images folder!")
        env.reset()

if __name__ == "__main__":
    _, instance = sys.argv
    gantt_class = GanttChart(instance)
    gantt_class.create_job_matrix()
    gantt_class.generate_gantt_csv()
    gantt_class.generate_chart()
    gantt_class.generate_gif()