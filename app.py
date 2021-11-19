# Core Pkg
import streamlit as st 

# TEXT
# title
st.title("Streamlit Crash Course")

# header/subheader
st.header("This is a header")

st.subheader("This is a subheader")

# text
st.text("This is so cool")

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
img = Image.open("example.jpeg")
st.image(img,width=300,caption="Streamlit Image")

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

# Area_chart
st.area_chart(df.head(20))
# Bar_chart

st.bar_chart(df.head(20))
# Line Chart

st.line_chart(df)

# Heatmap
c_plot = sns.heatmap(df.corr(),annot=True)
st.write(c_plot)
st.pyplot()

# Altair
# Vega
# D

# Date/Time
import datetime 
today = st.date_input("Today is",datetime.datetime.now())


import time
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

