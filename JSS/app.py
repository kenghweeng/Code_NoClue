import streamlit as st 
import pandas as pd
import numpy as np
import os
import cloudpickle as cp
import plotly.figure_factory as ff
import datetime
import base64
from PIL import Image

import matplotlib.pyplot as plt 
import matplotlib
matplotlib.use('Agg')

PAGE_CONFIG = {"page_title": "Code: No Clue", "page_icon": ":smiley:", "layout": "wide"}
st.set_page_config(**PAGE_CONFIG)

# TEXT
# title
html_temp = """
		<div style="background-color:{};padding:10px;border-radius:10px">
		<h1 style="color:white;text-align:center;">Reinforcement Learning on Patient Scheduling </h1>
		<h5 style="color:white;text-align:center;">Project: Code No Clue </h5>
		</div>
		<p>
		"""

# @st.cache(suppress_st_warning=True)
def load_image(img):
	im =Image.open(os.path.join(img))
	return im

def main():
	st.markdown(html_temp.format('royalblue'),unsafe_allow_html=True)

	menu = ["Home","For Staff", "For Patients"]
	sub_menu = ["Plot","Prediction","Metrics"]

	choice = st.sidebar.selectbox("Menu",menu)

	if choice == "Home":
		st.header("Project Background:")
		st.markdown("This application aims to realize the application of deep reinforcement learning to the healthcare setting, and was largely inspired by this particular [paper](https://arxiv.org/abs/2104.03760) written by Tassel et.al just this year! As you might have noticed, this project was named **Code No Clue**.\
			The name was intended to be a double entendre.")
		st.markdown("Firstly, it\'s meant to be a rhymy pun on the well-known term denoting an healthcare emergency: Code Blue. We adptly noted that in the healthcare setting, dealing with patients struck with dire conditions can be overwhelming and we may not know what the best \
				way to schedule the treatments for these patients could be. After talking to connections whom are currently working in the healthcare sector, we noted that the current scheduling procedure follows traditional, albeit, reasonable heuristics. Some of these heuristics \
				are (i) **FIFO -** **F**irst **I**n **F**irst **O**ut (more relevantly, First Come First Serve), (ii) **MWR -** Patient with **M**aximum **W**ork **R**emaining. We identified the opportunity to apply reinforcement-learning to learn a better scheduler for patient treatments.")

		st.markdown("Secondly, the name was meant to be a joke on how we tend to write code, which may or may not work but we have no idea why! :laughing:")
		
		st.header("Project Objective:")
		st.markdown("We illustrate a possible relevant use-case for our project: imagine if a healthcare facility is required to schedule various treatment procedures for a group of patients who have contracted COVID-19. For context, these treatments are necessary to ensure the well-being of these patients. \
			Hence, the healthcare facility is required to allocate the relevant medical personnel to administer the aforementioned treatment procedures. Some healthcare procedures we have identified, include:")
		
		st.write(
        """
		-   **Daily Nurse Rounds/Vital Checks**
		-   **X-Ray**
		-   **CT-Scan**
		-   **MRI**
		-   **Doctor Consultation**
		-   **Dispensary/Pharmacy**
		"""
		)

		st.markdown("To allow for a more realistic simulation for these respective procedures, we referred to a recent [medical journal](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7349722/pdf/healthcare-08-00077.pdf) written by Lee et.al during 2020. The paper quoted analytical simulations done in the Korean \
					medical setting, to model the processing time of these treatments with known distributions (normal and exponential). The implementation was done [here](https://github.com/kenghweeng/Code_NoClue/blob/presentation_env/JSS/covid_instance_generator.py), based on the reference above.")
		
		st.markdown("Recall that these patients have contracted COVID-19, and this implies the need for us to achieve the following objective. We want to ensure that all patients receive their respective medical treatments, while minimising the amount of time they interact with the medical staff.")
		st.markdown("We identified that this objective could be directly mapped to the well-known **Job-Scheduling Problem**, where the medical staff are considered '**machines**' and the patients are considered '**jobs**'. We want to minimise \
					the makespan of the entire system, which is equivalent to minimising the total duration all patients take to complete their treatments in our context. After understanding the background and objective, we now look at some possible simulations below with varying amounts of patients.")
		st.header("Application Features:")
		st.markdown('''
					With this application, you can now have a clue :smiley: 

					If you are a patient, you can look at your scheduled treatments along with an estimated queue status at a given timing. Feel free to go to the **"For Patients"** section in the sidebar.

					If you are a medical staff, you can look at the generated timetable for all patients along with the specific treatments that each patient is due to complete. Feel free to go to the **"For Staff"** section in the sidebar.
					''', unsafe_allow_html=True)

		st.header("Hospital Scheduling Simulation")
		patient_count = st.selectbox("Choose the number of patients to simulate:",["-- Choose --", 2,5,10,15,20])
		st.subheader("Example Input Format:")
		FIFO_GIF = f"https://github.com/kenghweeng/Code_NoClue/blob/presentation_env/JSS/images/covid{patient_count}_1_FIFO.gif?raw=true"
		RL_GIF = f"https://github.com/kenghweeng/Code_NoClue/blob/presentation_env/JSS/images/covid{patient_count}_1.gif?raw=true"
		if type(patient_count) == int:
			st.success(f"Looking at the simulation for {patient_count} patients:")
		with st.spinner("Waiting..."):
			import time
			time.sleep(3)
		if patient_count == 2:
			st.write(
				"""
				```
				2 5
				0 2 2 5 1 4 3 35 4 6
				0 3 2 18 3 33 1 9 4 14
				```
				"""
			)

		elif patient_count == 5:
			st.write(
				"""
				```
				5 5
				0 2 2 5 1 4 3 35 4 6
				0 3 2 18 3 33 1 9 4 14
				0 9 3 22 1 9 2 15 4 8
				0 3 1 8 3 29 2 13 4 20
				0 11 1 14 2 14 3 23 4 12
				```
				"""
			)
		
		elif patient_count == 10:
			st.write(
				"""
				```
				10 5
				0 2 2 5 1 4 3 35 4 6
				0 3 2 18 3 33 1 9 4 14
				0 9 3 22 1 9 2 15 4 8
				0 3 1 8 3 29 2 13 4 20
				0 11 1 14 2 14 3 23 4 12
				0 46 3 59 2 6 1 23 4 26
				0 1 3 22 1 14 2 16 4 11
				0 21 3 19 2 14 1 7 4 9
				0 1 3 22 1 17 2 22 4 13
				0 1 2 21 3 22 1 1 4 1
				```
				"""
			)
		elif patient_count == 15:
			st.write(
				"""
				```
				15 5
				0 2 2 5 1 4 3 35 4 6
				0 3 2 18 3 33 1 9 4 14
				0 9 3 22 1 9 2 15 4 8
				0 3 1 8 3 29 2 13 4 20
				0 11 1 14 2 14 3 23 4 12
				0 46 3 59 2 6 1 23 4 26
				0 1 3 22 1 14 2 16 4 11
				0 21 3 19 2 14 1 7 4 9
				0 1 3 22 1 17 2 22 4 13
				0 1 2 21 3 22 1 1 4 1
				0 12 3 12 1 3 2 15 4 12
				0 3 3 13 2 17 1 10 4 12
				0 31 3 46 2 13 1 13 4 16
				0 1 3 42 1 24 2 3 4 9
				0 4 1 27 2 2 3 12 4 17
				```
				"""
			)

		elif patient_count == 20:
			st.write(
				"""
				```
				20 5
				0 2 2 5 1 4 3 35 4 6
				0 3 2 18 3 33 1 9 4 14
				0 9 3 22 1 9 2 15 4 8
				0 3 1 8 3 29 2 13 4 20
				0 11 1 14 2 14 3 23 4 12
				0 46 3 59 2 6 1 23 4 26
				0 1 3 22 1 14 2 16 4 11
				0 21 3 19 2 14 1 7 4 9
				0 1 3 22 1 17 2 22 4 13
				0 1 2 21 3 22 1 1 4 1
				0 12 3 12 1 3 2 15 4 12
				0 3 3 13 2 17 1 10 4 12
				0 31 3 46 2 13 1 13 4 16
				0 1 3 42 1 24 2 3 4 9
				0 4 1 27 2 2 3 12 4 17
				0 13 2 10 3 29 1 4 4 14
				0 4 3 28 1 3 2 23 4 1
				0 9 1 4 3 9 2 18 4 5
				0 20 3 38 2 25 1 12 4 3
				0 8 3 27 2 9 1 11 4 8
				```
				"""
			)

		if type(patient_count) == int:
			labels = ['daily_rounds', 'x_ray', 'consultation', 'ct_scan', 'dispensary/pharmacy']
			machine2label = {i:j for i,j in enumerate(labels)}
			st.subheader("Mapping of Treatment Types to Names:")
			st.json(machine2label)
			st.markdown('''
					<u>**Explanation:**</u> <br><p>The first row contains 2 numbers, which are the number of patients and the number of treatment types respectively. Each of the following rows contains a sequence of **paired-values**: the treatment type and the corresponding treatment processing time for a given patient. The processing times are in minutes, and are given in the order of the treatments.</p>
					<p> For example, the treatments for Patient 1 is found in the second row,and they need to first go through Treatment Type 0 for 2 minutes. We can then interpret the next arranged treatment by taking a sliding window of 2 values to the right again.
					<p> For the sharp-eyed, note that each patient would begin Treatment Type 0 and conclude with Treatment Type 4. This corroborates to our understanding of a patient beginning with a registration/triage processes and ending the treatments with prescribed medication. In practice, we can estimate a patient's processing times at the various treatments given historical medical records.
						''', unsafe_allow_html=True)

			st.markdown('''
					<u>**What next?**</u> <br><p>After understanding the data specifications, proceed to `git clone` the repository found [here](https://github.com/kenghweeng/Code_NoClue/blob/presentation_env/JSS/). After cloning, remember to first install the relevant Python dependencies using the `requirements.txt` file provided. We recommend using `pyenv` or `poetry` to set up a virtual envrionment containing those dependencies. <br>
					
					After which, put the desired input format into a text (`.txt`) file and place it in the `/instances` folder of the cloned repository. We can then train a reinforcement-learning agent based on a popular policy-gradient method: the [PPO algorithm](https://openai.com/blog/openai-baselines-ppo/)! <br>
					
					Assuming you are in the directory containing `main.py`, run the following command and track the progress of reinforcement-learning agent using your favourite tool! We recommend Weights&Biases or the classic Tensorboard.</p>
						''', unsafe_allow_html=True)
			st.code("python3 main.py <filename of input>")
			st.markdown('''
						This would generate a solution sequence specifying the ordered timings of treatments for each patient, along with the total duration needed to complete all treatments for all patients. The solution is stored as a seralized `pickle` in the `/solutions` folder upon a successful training run. <br>

						Given the solution sequence, we can now generate fancy GIFs illustrating the chronological order of treatments for our patients. The following command is used to generate the relevant GIFs and static images, and stored in the folder `/images`:
						''', unsafe_allow_html=True)
			st.code("python3 generate_gantt.py <filename of input> # this assumes that the solution has been generated at /solutions") 
	
			st.subheader("Gantt chart for FIFO heuristic")
			st.markdown('''
						We first look at the GIF for the schedule generated by one of the traditional heuristics: the FIFO (First Patient In, First Patient Out) heuristic. The patients are ordered by their arrival times, and we see that they are scheduled for registration/triage processes on a first-come-first-serve basis.
						''', unsafe_allow_html=True)

			st.markdown(f"![FIFO for {patient_count} patients]({FIFO_GIF})")

			st.subheader("Gantt chart for RL heuristic")
			st.markdown('''
						We now look at the GIF for the schedule generated by the reinforcement-learning agent. Note that the schedule clearly does not follow the first-come-first-serve basis, and it tries to "intelligently" find the allocation which would reduce the overall make-span of the treatments. We also look at the solution generated 
						by the agent:
						''', unsafe_allow_html=True)
			st.markdown(f"![RL for {patient_count} patients]({RL_GIF})")
			
			if patient_count == 2:
				st.json({'solution': 
						[[ 3, 21, 26, 54, 89], 
						[ 0,  3, 21, 54, 63]], 
						'Total Duration': "95 minutes"})

			elif patient_count == 5:
				st.json({'solution': [[ 26,  44,  49, 116, 151],
						[ 23,  26,  60,  93, 102],
						[  0,   9,  31,  49,  64],
						[  9,  12,  31,  64,  77],
						[ 12,  53,  77,  93, 116]], 'Total Duration': "157 minutes"})
				
			
			elif patient_count == 10:
				st.json({'solution': [[ 13,  15,  40,  52,  95],
								[ 62,  96, 209, 242, 251],
								[  4, 187, 209, 218, 233],
								[  1,   4,  23,  62,  75],
								[ 66,  77, 114, 242, 265],
								[ 16,  87, 146, 152, 175],
								[ 15, 165, 187, 201, 217],
								[ 77, 146, 165, 179, 201],
								[  0,   1,  23,  40,  62],
								[ 65,  75, 265, 287, 288]], 'Total Duration': "289 minutes"}
				)
			
			elif patient_count == 15:
				st.json({'solution': [[  3,  24,  29, 322, 365],
									[127, 173, 357, 390, 399],
									[ 38,  77,  99, 158, 173],
									[130, 133, 145, 193, 206],
									[133, 188, 258, 287, 310],
									[ 81, 228, 287, 293, 322],
									[148, 206, 228, 242, 264],
									[  5, 187, 206, 220, 227],
									[  1,  43,  67,  84, 106],
									[  2,   3, 390, 412, 413],
									[ 26,  65,  84, 106, 121],
									[ 47, 174, 220, 242, 252],
									[ 50,  99, 145, 175, 188],
									[  0,   1,  43,  67,  70],
									[144, 148, 191, 310, 348]], 'Total Duration': "414 minutes"})

			elif patient_count == 20:
				st.json({'solution': [[  9,  26,  31, 199, 234],
									[189, 295, 488, 521, 530],
									[111, 466, 488, 497, 512],
									[192, 195, 287, 316, 329],
									[100, 145, 329, 343, 373],
									[122, 366, 425, 440, 467],
									[121, 136, 172, 246, 268],
									[168, 256, 343, 357, 364],
									[120, 234, 256, 273, 304],
									[  4,   5, 521, 543, 544],
									[ 88, 425, 437, 440, 455],
									[  1, 158, 229, 246, 256],
									[ 57,  90, 136, 159, 174],
									[  0,   1,  43,  67,  70],
									[ 11, 118, 149, 275, 287],
									[ 35,  48, 437, 466, 493],
									[  5, 171, 203, 206, 229],
									[ 48,  67,  81, 151, 169],
									[ 15,  43,  81, 106, 118],
									[195, 316, 357, 366, 385]], 'Total Duration': "545 minutes"})

		

# 		# MARKDOWN
# 		st.markdown("# This is markdown")


# 		# Boostrap Alert/Color Text
# 		st.success("Succcess!")

# 		st.info("Information")
# 		st.warning("This is a warning")
# 		st.error("This is an error")

# 		st.exception('NameError()')

# 		# MEDIA
# 		# Images
# 		# img = Image.open("images/covid2_1.png")
# 		st.image("https://github.com/kenghweeng/Code_NoClue/blob/presentation_env/JSS/images/covid2_1.png?raw=true")
# 		"""### gif from url"""
# 		st.markdown("![Alt Text](https://github.com/kenghweeng/Code_NoClue/blob/presentation_env/JSS/images/covid2_1.gif?raw=true)")


# # 		st.markdown(
# # 			f'<img src="data:image/gif;base64,{data_url}" alt="cat gif">',
# # 			unsafe_allow_html=True,
# # )

# 		# # Audio
# 		# audio_file = open('example.mp3',"rb")
# 		# audio_bytes = audio_file.read()
# 		# st.audio(audio_bytes,format="audio/mp3")

# 		# # Video
# 		# video_file = open("example.mp4","rb")
# 		# video_bytes = video_file.read()
# 		# st.video(video_bytes)


# 		# Video URL/YTB
# 		# st.video("https://www.youtube.com/watch?v=_9WiB2PDO7k")


# 		# WIDGET
# 		st.button("Submit")

# 		if st.button("Play"):
# 			st.text("Hello world")

# 		# Checkbox
# 		if st.checkbox("Show/hide"):
# 			st.success("Hiding or Showing")

# 		# Radio
# 		gender = st.radio("Your Gender",["Male","Female"])

# 		if gender == 'Male':
# 			st.info("Is a male")

# 		# Select
# 		location = st.selectbox("Your Location",["UK","USA","India","Accra"])


# 		# Multiselect
# 		occupation = st.multiselect("Your Occupation",["Developer","Doctor","BusinessMan","Banker"])


# 		# TEXT INPUT
# 		name = st.text_input("Your Name","Type Here")
# 		st.text(name)

# 		# NUMBER INPUT
# 		age = st.number_input("Age")

# 		# TEXT_AREA
# 		message = st.text_area("Your Message","Type here")

# 		# SLider
# 		level = st.slider("Your Level",2,6)

# 		# Balloons
# 		# st.balloons()



# 		# DATA SCIENCE
# 		st.write(range(10))

# 		#DATAFRAME

# 		import pandas as pd 
# 		df = pd.read_csv("iris.csv")

# 		#M1
# 		st.dataframe(df.head())
# 		# #M2
# 		# st.write(df.head())


# 		# TABLES

# 		st.table(df.head())
# 		# PLOT

# 		# Plot Pkgs
# 		import matplotlib.pyplot as plt 
# 		import seaborn as sns 

# 		# # Area_chart
# 		# st.area_chart(df.head(20))
# 		# # Bar_chart

# 		# st.bar_chart(df.head(20))
# 		# # Line Chart

# 		# st.line_chart(df)

# 		# Heatmap
# 		# c_plot = sns.heatmap(df.corr(),annot=True)
# 		# st.write(c_plot)
# 		# st.pyplot()

# 		# Altair
# 		# Vega
# 		# D

# 		# Date/Time
# 		today = st.date_input("Today is",datetime.datetime.now())


# 		the_time = st.time_input("The time is",datetime.time(10,0))


# 		# Display JSON,CODE
# 		data = {"name":"John","salar":50000}
# 		st.json(data)

# 		# Display Code
# 		st.code("import numpy as np")

# 		st.code("import numpy as np",language='python')

# 		julia_code ="""
# 		function doit(num::int)
# 			println(num)
# 		end
# 		"""

# 		st.code(julia_code,language='julia')

# 		# with st.echo():	# show code block.
# 		# 	# This is a comment
# 		# 	import textblob

# 		# Progressbar
# 		import time 
# 		my_bar = st.progress(0)
# 		for p in range(10):
# 			my_bar.progress(p+1)

# 		# Spinner

# 		with st.spinner("Waiting..."):
# 			time.sleep(5)
# 		st.success("Finished")

	elif choice == "For Staff":
		st.subheader("Welcome back, Staff!")
		patient_count = st.selectbox("Patient Count:", ["-- Select --", 2,5,10,15,20])
		with st.spinner("Waiting..."):
			import time
			time.sleep(2)
		if type(patient_count) == int:
			st.success("Simulation for {} patients completed".format(patient_count))
			st.subheader("Here is your timetable for today:")
			st.image(f"https://github.com/kenghweeng/Code_NoClue/blob/presentation_env/JSS/images/covid{patient_count}_1.png?raw=true")
			st.subheader("Here are the details for all patients:")
			st.write("Click on the column headers to toggle between sorted ascending/descending order")
			df = pd.read_csv(f"https://github.com/kenghweeng/Code_NoClue/blob/presentation_env/JSS/schedules/covid{patient_count}_1.csv?raw=true")
			st.dataframe(df)
			st.subheader("Which patients do you want to look at?")

			# Multiselect
			patients_lst = st.multiselect("Patients:",[f"Patient {i}" for i in range(1,patient_count+1)])
			if len(patients_lst) > 0:
				st.info(f"You have chosen: {', '.join(patients_lst)}")
				st.dataframe(df[df["Task"].isin(patients_lst)])

	elif choice == "For Patients":
		st.subheader("Now we have a clue! :smiley:")
		patient_count = st.selectbox("Patient Count:", ["-- Select --", 2,5,10,15,20])
		with st.spinner("Waiting..."):
			import time
			time.sleep(2)

		if type(patient_count) == int:
			st.success("Simulation for {} patients completed".format(patient_count))
			df = pd.read_csv(f"https://github.com/kenghweeng/Code_NoClue/blob/presentation_env/JSS/schedules/covid{patient_count}_1.csv?raw=true")
			df["Start"] = pd.to_datetime(df["Start"], format="%H:%M:%S").dt.time

			patient_name = st.selectbox("You are:", ["None"] + [f"Patient {i}" for i in range(1,patient_count+1)])
			if patient_name != "None":
				st.info(f"Here is your schedule for today, {patient_name}")
				st.subheader("Your timetable:")
				st.write("Click on the column headers to toggle between sorted ascending/descending order")
				patient_df = df.loc[df["Task"] == patient_name]
				
				patient_df["Start"] = patient_df["Start"].apply(lambda x: x.strftime("%H:%M:%S"))
				st.dataframe(patient_df)
				st.subheader("Queue status at time:")
				the_time = st.time_input("The time is",datetime.time(0,0))
				curr_treatment = st.selectbox("What treatment are you waiting for:", ["None"] + list(set(df["Resource"].values)))
				
				if curr_treatment != "None":
					treatment_df = df.loc[df["Resource"] == curr_treatment]
					patient_time = treatment_df.loc[treatment_df["Task"] == patient_name, "Start"].values[0]
					st.subheader("People before you:")
					filtered = treatment_df.loc[(treatment_df["Start"]>the_time) & (treatment_df["Start"]<patient_time), ["Task", "Finish", "Resource"]].sort_values("Finish")
					st.dataframe(filtered)
					if filtered.shape[0] == 0:
						st.success("You are the first in line! It will be your turn soon.")
					else:
						st.write(f"Estimated: {filtered.shape[0]} people before you")
					



			





		

if __name__ == '__main__':
	main()
