import streamlit as st  #Web App
import numpy as np #Image Processing 
import torch
import matplotlib.pyplot as plt

from PIL import Image
from cnn import Net


#title
st.title("CIFAR10 image classification")

#image uploader
image = st.file_uploader(label = "Upload your image in CIFAR-10 format here",type=['png','jpg','jpeg'])

PATH = './cifar_net.pth'

@st.cache
def load_model(): 
    net = Net()
    net.load_state_dict(torch.load(PATH))
    return net

net = load_model() #load model

if image is not None:

    # st.image(Image.open(image)) ## display image

    input_image = plt.imread(image)

    input_image = torch.Tensor(np.reshape(
        input_image,
        newshape = (1, -1, input_image.shape[0], input_image.shape[1])
    ))

    # st.image(input_image.item()) #display image

    with st.spinner("🤖 AI is at Work! "):
        

        outputs = net(input_image) 

        values, indices = torch.max(outputs, dim=1)

        # Define list of classes
        classes = ('plane', 'car', 'bird', 'cat',
                'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

        result_text = ' '.join(f'{classes[indices[j]]:5s}' for j in range(indices.shape[0]))

        st.write(result_text)
    
    st.balloons()
else:
    st.write("Upload an Image")

st.caption("Made by CS. ")
