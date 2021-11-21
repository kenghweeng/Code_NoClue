import streamlit as st 
import pandas as pd
import numpy as np
import os
import pickle
import plotly.figure_factory as ff
import datetime
import time
import base64

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
		# talk about objective, makespan, Map to JSS, patients, resource groups,

		st.header("Data Specifications:")
		
		st.header("Plots")

		# MARKDOWN
		st.markdown("# This is markdown")

		# Links
		st.markdown("[Google](https://google.com)")

		url_link = "https://jcharistech.com"

		st.markdown(url_link)

		# Custom Color/Style
		# html_page = """
		# <div style="background-color:tomato;padding:50px">
		# 	<p style="font-size:50px">Streamlit is Awesome</p>
			
		# </div>
		# """
		# st.markdown(html_page,unsafe_allow_html=True)

		# html_form = """
		# <div>
		# 	<form>
		# 	<input type="text" name="firstname"/>

		# 	</form>
		# </div>
		# """

		# st.markdown(html_form,unsafe_allow_html=True)


		# Boostrap Alert/Color Text
		st.success("Succcess!")

		st.info("Information")
		st.warning("This is a warning")
		st.error("This is an error")

		st.exception('NameError()')

		# MEDIA
		# Images
		from PIL import Image
		img = Image.open("images/covid2_1.png")

		"""### gif from local file"""
		file_ = open("images/covid2_1.gif", "rb")
		contents = file_.read()
		data_url = base64.b64encode(contents).decode("utf-8")
		file_.close()

		st.markdown(
			f'<img src="data:image/gif;base64,{data_url}" alt="cat gif">',
			unsafe_allow_html=True,
)
		st.image(img,caption="Streamlit Image")

		# # Audio
		# audio_file = open('example.mp3',"rb")
		# audio_bytes = audio_file.read()
		# st.audio(audio_bytes,format="audio/mp3")

		# # Video
		# video_file = open("example.mp4","rb")
		# video_bytes = video_file.read()
		# st.video(video_bytes)


		# Video URL/YTB
		# st.video("https://www.youtube.com/watch?v=_9WiB2PDO7k")


		# WIDGET
		st.button("Submit")

		if st.button("Play"):
			st.text("Hello world")

		# Checkbox
		if st.checkbox("Show/hide"):
			st.success("Hiding or Showing")

		# Radio
		gender = st.radio("Your Gender",["Male","Female"])

		if gender == 'Male':
			st.info("Is a male")

		# Select
		location = st.selectbox("Your Location",["UK","USA","India","Accra"])


		# Multiselect
		occupation = st.multiselect("Your Occupation",["Developer","Doctor","BusinessMan","Banker"])


		# TEXT INPUT
		name = st.text_input("Your Name","Type Here")
		st.text(name)

		# NUMBER INPUT
		age = st.number_input("Age")

		# TEXT_AREA
		message = st.text_area("Your Message","Type here")

		# SLider
		level = st.slider("Your Level",2,6)

		# Balloons
		# st.balloons()



		# DATA SCIENCE
		st.write(range(10))

		#DATAFRAME

		import pandas as pd 
		df = pd.read_csv("iris.csv")

		#M1
		st.dataframe(df.head())
		# #M2
		# st.write(df.head())


		# TABLES

		st.table(df.head())
		# PLOT

		# Plot Pkgs
		import matplotlib.pyplot as plt 
		import seaborn as sns 

		# # Area_chart
		# st.area_chart(df.head(20))
		# # Bar_chart

		# st.bar_chart(df.head(20))
		# # Line Chart

		# st.line_chart(df)

		# Heatmap
		# c_plot = sns.heatmap(df.corr(),annot=True)
		# st.write(c_plot)
		# st.pyplot()

		# Altair
		# Vega
		# D

		# Date/Time
		today = st.date_input("Today is",datetime.datetime.now())


		the_time = st.time_input("The time is",datetime.time(10,0))


		# Display JSON,CODE
		data = {"name":"John","salar":50000}
		st.json(data)

		# Display Code
		st.code("import numpy as np")

		st.code("import numpy as np",language='python')

		julia_code ="""
		function doit(num::int)
			println(num)
		end
		"""

		st.code(julia_code,language='julia')

		# with st.echo():	# show code block.
		# 	# This is a comment
		# 	import textblob

		# Progressbar
		import time 
		my_bar = st.progress(0)
		for p in range(10):
			my_bar.progress(p+1)

		# Spinner

		with st.spinner("Waiting..."):
			time.sleep(5)
		st.success("Finished")

	elif choice == "My Schedule":
		st.subheader("Now you can have a clue!")


if __name__ == '__main__':
	main()
