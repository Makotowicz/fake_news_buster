import streamlit as st
import pandas as pd
from preprocess_pipeline_1d import final_call
import time
import joblib

# st.beta_set_page_config(
#      page_title="test",
#      page_icon=":brokkoli:",
#      layout="centered",
#      initial_sidebar_state="collapsed"
#  )

html_temp = """

<div style="background-color:black; padding:10px">
<h1 style="color:white;font-size:xxx-large; text-align:center;">📰Fake News Detector</h1>

<div>
"""

st.markdown(html_temp.format(), unsafe_allow_html=True)


CSS = """
body {
    background-image: url("https://images.unsplash.com/photo-1566378246598-5b11a0d486cc?ixlib=rb-1.2.1&ixid=eyJhcHBfaWQiOjEyMDd9&auto=format&fit=crop&w=1268&q=80");
    background-size: cover;
}
"""

# CSS = """
# h1 {
#     color: red;
#     font-weight: bold;
#     font-size: large;
# }
# body {
#     background-image: url("https://images.unsplash.com/photo-1566378246598-5b11a0d486cc?ixlib=rb-1.2.1&ixid=eyJhcHBfaWQiOjEyMDd9&auto=format&fit=crop&w=1268&q=80");
#     background-size: cover;
#     font-size: large;
# }
# """


st.write(f'<style>{CSS}</style>', unsafe_allow_html=True)


# st.title("""Fake News Detection
# ##
# """)

st.subheader("")

from PIL import Image
image = Image.open('Download.jfif')
st.image(image, use_column_width=False)


# '''Input Box for TITLE '''
#st.subheader("Please input the title of the article:")

html_temp3= """

<div style="background-color:black;padding:5px">
<h1 style="color:white; font-size:x-large; text-align:left;">🕵️‍♂️ Please input title:</h1>

<div>
"""


st.markdown(html_temp3.format(), unsafe_allow_html=True)

title = st.text_input(" ")
# st.write("Title:", title)
if not title:

    st.stop()
    #st.error('Please enter the title of the article')
# else:
#     st.success("Thanks for inputting the title of the article")


# '''Input Box for BODY '''
#st.subheader("Please input the body of the article:")
html_temp4= """

<div style="background-color:black;padding:5px">
<h1 style="color:white; font-size:x-large; text-align:left;">🕵️‍♂️ Please input body:</h1>

<div>
"""

st.markdown(html_temp4.format(), unsafe_allow_html=True)

text = st.text_area("   ")

if not text:
    #st.error('Please enter the body of the article')
    st.stop()

st.write("Text consists of:", len(text), "number of characters")


# '''Input Box for DATE '''
#st.subheader("Please input the date the article was published:")
html_temp5= """

<div style="background-color:black;padding:5px">
<h1 style="color:white; font-size:x-large; text-align:left;">🕵️‍♂️ Please input date:</h1>

<div>
"""
st.markdown(html_temp5.format(), unsafe_allow_html=True)

date = st.text_input("Format needed: YYYY-MM-DD")
# st.write("Date:", text)
if not date:
    #st.error('Please enter the publication date of the article')
    st.stop()

# if title and text and date not empty:
#     st.success("Thanks for inputting all relevant information")


inputs = {'title': [title],
         'text': [text],
         'date': [date]}

df = pd.DataFrame(inputs, columns = ['title', 'text', 'date'])
df['date'] = pd.to_datetime(df['date'])

with st.spinner('Wait for it...'):
    time.sleep(3)

df = final_call(df)

st.success('Done!')


loaded_model = joblib.load("Makotowicz_buzzfeed.sav")
result = loaded_model.predict_proba(df)*100
result = ("{:.0f}".format(result[0][1]))


#st.write(f'We deem the probability that the article is fake to be ', result)
if int(result) < 50:
    html_design = """

    <div style="background-color:black; padding:10px">
    <h2 style="color:white;font-size:x-large;font-weight:bold; text-align:center">✔️ Probability that the article is fake to be {} %</h2>

    <div>
    """

    st.markdown(html_design.format(result), unsafe_allow_html=True)

else:
    html_design = """

    <div style="background-color:black; padding:10px">
    <h2 style="color:white;font-size:x-large;font-weight:bold; text-align:center">⚠️ Probability that the article is fake to be {} %</h2>

    <div>
    """

    st.markdown(html_design.format(result), unsafe_allow_html=True)
# def local_css(file_name):
#     with open(file_name) as f:
#         st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
# #def remote_css(url):
# #    st.markdown(f’<link href=“{url}” rel=“stylesheet”>’, unsafe_allow_html=True)
# local_css('style.css')




