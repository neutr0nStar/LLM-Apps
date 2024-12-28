from dotenv import load_dotenv
from together import Together
import streamlit as st

# 
#   TOGETHER AI
# 

load_dotenv()
client = Together()

def generate_image(prompt: str):
    res = client.images.generate(
        prompt=prompt,
        model="black-forest-labs/FLUX.1-schnell-Free",
        steps=4,
        height=768,
        width=1024
    )

    return res.data[0].url


#
#   STREAMLIT APP
#

st.set_page_config(page_title="AI Image Generator", layout="wide")

# Session state to store result image url
if 'res_img' not in st.session_state:
    st.session_state['res_img'] = None

# 2 column layout
column1, column2 = st.columns([2, 3]) # Column widths in ration 2:3

with column1:

    st.write("## FLUX.1 [schnell] Image Generator")

    # Prompt
    prompt = st.text_area(label="Enter prompt")

    # submit button
    submit = st.button(label="Submit", type="primary", use_container_width=True)

    if submit:
        # On submit, generate image and display it
        st.session_state['res_img'] = generate_image(prompt)
        st.rerun()

with column2:
    # Show result image
    if st.session_state['res_img']:
        st.image(st.session_state['res_img'])