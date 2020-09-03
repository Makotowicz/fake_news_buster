import streamlit as st
import pandas as pd
from preprocess_pipeline_final import final_call
import time
import joblib


html_temp = """

<div style="background-color:white">
<h1 style="color:black;font-size:xxx-large; text-align:center;font-family:sans-serif;">Fake News Detector</h1>

<div>
"""

st.markdown(html_temp.format(), unsafe_allow_html=True)


CSS = """
body {
    background-image: url("https://images.unsplash.com/photo-1495020689067-958852a7765e?ixlib=rb-1.2.1&auto=format&fit=crop&w=400&q=60");
    background-size: cover;
}
"""

st.write(f'<style>{CSS}</style>', unsafe_allow_html=True)


st.subheader("")
# '''Input Box for TITLE '''
#st.subheader("Please input the title of the article:")

html_temp3= """

<div style="background-color:white;padding:5px">
<h1 style="color:black; font-size:x-large; text-align:left;">➡️ Please input title:</h1>

<div>
"""


st.markdown(html_temp3.format(), unsafe_allow_html=True)


title = st.text_input(" ")
# st.write("Title:", title)
if not title:

    st.stop()

html_temp4= """

<div style="background-color:white;padding:5px">
<h1 style="color:black; font-size:x-large; text-align:left;">➡️ Please input body:</h1>

<div>
"""

st.markdown(html_temp4.format(), unsafe_allow_html=True)


text = st.text_area("   ")

if not text:
    st.stop()



# '''Input Box for DATE '''
#st.subheader("Please input the date the article was published:")
html_temp5= """

<div style="background-color:white;padding:5px">
<h1 style="color:black; font-size:x-large; text-align:left;">➡️ Please input date (yyyy-mm-dd):</h1>

<div>
"""
st.markdown(html_temp5.format(), unsafe_allow_html=True)

date = st.text_input("  ")
# st.write("Date:", text)
if not date:
    #st.error('Please enter the publication date of the article')
    st.stop()


inputs = {'title': [title],
         'text': [text],
         'date': [date]}

df = pd.DataFrame(inputs, columns = ['title', 'text', 'date'])
df['date'] = pd.to_datetime(df['date'])


if st.button("Let's investigate!"):
    result_1 = final_call(df)

    html_temp7= """

    <div style="background-color:rgb(100,149,237,0.75);">
    <h1 style="color:black; font-size:large; text-align:center;"> The model is running...</h1>

    <div>
    """
    st.markdown(html_temp7.format(), unsafe_allow_html=True)
    time.sleep(4)

    html_temp10= """

    <div style="background-color:rgb(100,149,237,0.75);">
    <h1 style="color:black; font-size:large; text-align:center;"> Investigation done</h1>

    <div>
    """
    st.markdown(html_temp10.format(), unsafe_allow_html=True)
    # st.success('Done!')
    time.sleep(2)
    result_1 = ("{:.2f}".format(result_1[0][1]*100))
# with st.spinner('Wait for it...'):
#     time.sleep(3)



# loaded_model = joblib.load("Makotowicz_buzzfeed.sav")
# result = loaded_model.predict_proba(df)*100
# result = ("{:.0f}".format(result[0][1]))



#st.write(f'We deem the probability that the article is fake to be ', result)
    if float(result_1) < 50:
        html_design = """

        <div style="background-color:rgb(152,251,152,0.75); padding:10px">
        <h2 style="color:black;font-size:x-large;font-weight:bold; text-align:center">✔️ The probability that the article is fake is {} %</h2>

        <div>
        """

        st.markdown(html_design.format(result_1), unsafe_allow_html=True)

    else:
        html_design = """

        <div style="background-color:rgb(240,128,128,0.75); padding:10px">
        <h2 style="color:black;font-size:x-large;font-weight:bold; text-align:center">⚠️ The probability that the article is fake is {} %</h2>

        <div>
        """

        st.markdown(html_design.format(result_1), unsafe_allow_html=True)

        st.subheader("")

        # from PIL import Image
        # image = Image.open('Download.jfif')
        # st.image(image, use_column_width=False)




