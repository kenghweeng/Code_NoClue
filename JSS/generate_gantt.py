import datetime
import pandas as pd
import gym
import numpy as np
import plotly.figure_factory as ff
import sys
import pickle
import random
import imageio

# refactor to expose method for generate CSV, then from CSV manipulate solution seq for rendering GIFs

class GanttChart:
    def __init__(self, instance):
        self.instance = instance
        self.start_timestamp = datetime.datetime.now().timestamp()
        with open(f'solutions/{self.instance}_sol.pkl', 'rb') as f:
            self.solution = pickle.load(f)
        
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

    def generate_from_timings(self):
        df = []
        self.colors = [
            tuple([random.random() for _ in range(3)]) for _ in range(self.machines)
        ]
        for job in range(self.jobs):
            i = 0
            while i < self.machines and self.solution[job][i] != -1:
                dict_op = dict()
                dict_op["Task"] = 'Job {}'.format(job)
                dict_op["Task_Num"] = job
                start_sec = self.start_timestamp + self.solution[job][i]
                finish_sec = start_sec + self.instance_matrix[job][i][1]
                dict_op["Start"] = datetime.datetime.fromtimestamp(start_sec)
                dict_op["Finish"] = datetime.datetime.fromtimestamp(finish_sec)
                dict_op["Resource"] = "Machine {}".format(self.instance_matrix[job][i][0])
                dict_op["Resource_Num"] = self.instance_matrix[job][i][0]
                
                df.append(dict_op)
                i += 1

        fig = None
        if len(df) > 0:
            df = pd.DataFrame(df)

            df.sort_values('Resource_Num', inplace=True)
            df_grouped = df.groupby("Resource")
            for name, group in df_grouped:
                print(group.sort_values("Start"))
                group.sort_values("Start", inplace=True)
                print([job for job in group['Task'].values])
                print([int(job.split(' ')[-1]) for job in group['Task'].values])
            
            fig = ff.create_gantt(df, index_col='Resource', colors=self.colors, show_colorbar=True,
                                  group_tasks=True)
            fig.update_yaxes(autorange="reversed")  # otherwise tasks are listed from the bottom up
        
        print("rendering")
        temp_image = fig.to_image()
        imageio.imsave('test.png', imageio.imread(temp_image))

    # def generate_from_sequence(self):

if __name__ == "__main__":
    _, instance = sys.argv
    gantt_class = GanttChart(instance)
    gantt_class.create_job_matrix()
    gantt_class.generate_from_timings()



