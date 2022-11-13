
import streamlit as st
from PIL import Image
import subprocess
import os

# Define the function to save the image as number.png
def save_image(uploadedfile):
  with open(os.path.join("LeNet5\images","number.png"),"wb") as f:
     f.write(uploadedfile.getbuffer())


st.title("LeNet-5")

st.text("""LeNet-5 is a convolutional neural network (CNN) developed by Yann LeCun in 1998.
It was the first CNN to be used in the field of computer vision.
It was used to classify handwritten digits and images of traffic signs.
It aachieved state of the art performance for its time.
It has 2 convolutional layers and 3 fully connected layers.""")

image = Image.open('LeNet5\\architecture.png')
st.image(image, caption='The architecture of LeNet-5')

st.header('Try it out!')

st.text("Upload your image of a handwritten digit and the model will identify it...")

uploaded_file = st.file_uploader("Choose an image...", type="png")



if uploaded_file is not None:
    # save the image
    save_image(uploaded_file)
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    with st.spinner('Identifying...'):
        # print the output of predict.py
        st.write("LeNet thinks this digit is: ",int(subprocess.check_output(["python", "LeNet5\predict.py"])))

