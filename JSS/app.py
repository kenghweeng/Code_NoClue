import streamlit as st 
import pandas as pd
import numpy as np
import os
import pickle
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


@st.cache
def load_image(img):
	im =Image.open(os.path.join(img))
	return im

def main():
	st.markdown(html_temp.format('royalblue'),unsafe_allow_html=True)

	menu = ["Home","My Schedule"]
	sub_menu = ["Plot","Prediction","Metrics"]

	choice = st.sidebar.selectbox("Menu",menu)

	if choice == "Home":
		st.header("Project Background:")
		st.markdown("This application aims to realize the application of deep reinforcement learning to the healthcare setting, and was largely inspired by this particular [paper](https://arxiv.org/abs/2104.03760) written by Tassel et.al just this year! As you might have noticed, this project was named **Code No Clue**.\
			The name was intended to be a double entendre.")
		st.markdown("Firstly, it\'s meant to be a rhymy pun on the well-known term denoting an healthcare emergency: Code Blue. We adptly noted that in the healthcare setting, dealing with patients struck with dire conditions can be overwhelming and we may not know what the best \
				way to schedule the treatments for these patients could be. After talking to connections whom are currently working in the healthcare sector, we noted that the current scheduling procedure follows traditional, albeit, reasonable heuristics. Some of these heuristics \
				are (i) **FIFO -** **F**irst **I**n **F**irst **O**ut (more relevantly, First Come First Serve), (ii) **MWKR -** Patient with **M**aximum **W**or**k** **R**emaining. We identified the opportunity to apply reinforcement-learning to learn a better scheduler for patient treatments.")

		st.markdown("Secondly, the name was meant to be a joke on how we tend to write code, which may or may not work but we have no idea why! :laughing:")
		
		st.header("Project Objective:")
		st.markdown("We illustrate a possible relevant use-case for our project: imagine if a healthcare facility is required to schedule various treatment procedures for a group of patients who have contracted COVID-19. For context, these treatments are necessary to ensure the well-being of these patients. \
			Hence, the healthcare facility is required to allocate the relevant medical personnel to administer the aforementioned treatment procedures. Some healthcare procedures we have identified, include:")
		
		st.write(
        """
		-   **Registration/Triage**
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
					the makespan of the entire system, which is equivalent to minimising the total duration all patients take to complete their treatments in our context. After understanding the background and objective, we now look at some possible simulations below.")

		st.header("Hospital Scheduling Simulation")
		patient_count = st.selectbox("Number of patients",["-- Choose --", 2,5,10,15,20])
		FIFO_GIF = f"https://github.com/kenghweeng/Code_NoClue/blob/presentation_env/JSS/images/covid{patient_count}_1_FIFO.gif?raw=true"
		RL_GIF = f"https://github.com/kenghweeng/Code_NoClue/blob/presentation_env/JSS/images/covid{patient_count}_1.gif?raw=true"
		with st.spinner("Waiting..."):
			import time
			time.sleep(3)
		st.subheader("Example Input Format:")
		if patient_count == 2:
			st.success(f"Looking at the simulation for {patient_count} patients:")
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
			st.success(f"Looking at the simulation for {patient_count} patients:")
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
			st.success(f"Looking at the simulation for {patient_count} patients:")
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
			st.success(f"Looking at the simulation for {patient_count} patients:")
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
			st.success(f"Looking at the simulation for {patient_count} patients:")
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

		
		st.text("Explanation")
		st.subheader("Gantt chart for FIFO heuristic")
		st.markdown(f"![FIFO for {patient_count} patients]({FIFO_GIF})")

		st.subheader("Gantt chart for RL heuristic")
		st.markdown(f"![RL for {patient_count} patients]({RL_GIF})")
		

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

	elif choice == "My Schedule":
		st.subheader("Now you can have a clue!")

if __name__ == '__main__':
	main()
