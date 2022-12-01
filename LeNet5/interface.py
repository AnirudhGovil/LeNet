
import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import subprocess
import os

# Define the function to save the image as number.png
def save_image(uploadedfile):
  with open(os.path.join("images","number.png"),"wb") as f:
     f.write(uploadedfile.getbuffer())

def save_drawn_image(uploadedfile):
  im = Image.fromarray(uploaded_file)
  im.save(os.path.join("images","number.png"))


st.title("LeNet-5")

st.text("""LeNet-5 is a convolutional neural network (CNN) developed by Yann LeCun in 1998.
It was the first CNN to be used in the field of computer vision.
It was used to classify handwritten digits and images of traffic signs.
It achieved state of the art performance for its time.
It has 2 convolutional layers and 3 fully connected layers.""")

image = Image.open('architecture.png')
st.image(image, caption='The architecture of LeNet-5')

st.header('Try it out!')

st.text("Use your mouse to draw a digit and the model will identify it...")

# uploaded_file = st.file_uploader("Choose an image...", type="png")

# if uploaded_file is not None:
#     # save the image
#     save_image(uploaded_file)
#     image = Image.open(uploaded_file)
#     st.image(image, caption='Uploaded Image.', use_column_width=True)
#     with st.spinner('Identifying...'):
#         # print the output of predict.py
#         st.write("LeNet thinks this digit is: ",int(subprocess.check_output(["python", "LeNet5\predict.py"])))


# Create a canvas component
canvas_result = st_canvas(
    fill_color="White",  # Fixed fill color with some opacity
    stroke_color="White",
    stroke_width=15,
    update_streamlit=True,
    width=160,
    height=160,
    key="canvas",
)

# Do something interesting with the image data and paths
if canvas_result.image_data is not None:
  option = st.selectbox( 'Select Classifier', ('LeNet 5', 'LeNet 1', 'LeNet 4', 'Booseted LeNet 4', 'One-Hidden-Layer FCMNN', 'Two-Hidden-Layer FCMNN', 'KNN', 'Linear Classifier', 'Polynomial Classifier', 'RBF Network', 'SVM'))
  if st.button('Predict'):
    st.write("option is: ",option)
    uploaded_file = canvas_result.image_data
    if uploaded_file is not None:
    # save the image
      save_drawn_image(uploaded_file)
      with st.spinner('Identifying...'):
          # print the output of predict.py
          if(option == 'LeNet 5'):
            st.write("LeNet thinks this digit is: ",int(subprocess.check_output(["python", "LeNet_5.py"])))
          elif(option == 'KNN'):
            st.write("LeNet thinks this digit is: ",int(subprocess.check_output(["python", "KNN.py"])))
          elif(option == 'Linear Classifier'):
            st.write("LeNet thinks this digit is: ",int(subprocess.check_output(["python", "Linear_Classifier.py"])))
          elif(option == 'Polynomial Classifier'):  
            st.write("LeNet thinks this digit is: ",int(subprocess.check_output(["python", "Polynomial_Classifier.py"])))
          elif(option == 'RBF Network'):
            st.write("LeNet thinks this digit is: ",int(subprocess.check_output(["python", "RBF_Network.py"])))
          elif(option == 'SVM'):
            st.write("LeNet thinks this digit is: ",int(subprocess.check_output(["python", "SVM.py"])))


