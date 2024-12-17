from dotenv import load_dotenv
from together import Together
import base64
import streamlit as st

#
#   TOGETHER AI 
#
load_dotenv()
client = Together()

def get_vision_inference(image_bytes: bytes, prompt: str):
    # Process image bytes
    image_bytes = base64.b64encode(image_bytes).decode('utf-8')
    
    # Generate stream
    stream = client.chat.completions.create(
        model="meta-llama/Llama-Vision-Free",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url", 
                        "image_url":{ "url": f"data:image;base64,{image_bytes}"}
                    },
                ],
            }
            ],
        stream=True,
        )
    
    for chunk in stream:
        if chunk.choices:
            yield chunk.choices[0].delta.content


#
#   STREAMLIT APP
#

# Switch to wide layout to accomode 2 columns
st.set_page_config(layout="wide", page_title="Llama Vision Inference")

st.title("Llama-vision Vision Inferrence")

# We'll use a 2 column layout
column1, column2 = st.columns(2)

with column1:
    st.write("#### Step 1. Upload image")

    # File uploader widget
    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    # Display uploaded image
    if uploaded_image is not None:
        st.image(uploaded_image, caption="Uploaded Image", use_container_width=True)

with column2:
    st.write("#### Step 2: Write prompt and submit")
    
    # Prompt input
    prompt = st.text_input("Enter prompt")

    # Submit button
    submit = st.button("Submit", type="primary", use_container_width=True)

    if submit:
        if not uploaded_image:
            st.error("Image required")
        elif not prompt:
            st.error("Prompt required")
        else:
            with st.chat_message("ai"):
                st.write_stream(get_vision_inference(uploaded_image.getvalue(), prompt))