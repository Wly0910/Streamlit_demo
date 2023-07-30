import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image
import utils
import os
import style2
import nibabel as nib
import io

st.title('Brain Tumor Segmentation')

# select MRI pattern
pattern_name = st.sidebar.selectbox(
    'Select Your MRI',
    ('t1', 't1ce', 'flair', 't2')
)

# select model
model_name = st.sidebar.selectbox(
    'Select Model',
    tuple(os.listdir("./saved_models"))
)

# select patient
patient_name = st.sidebar.selectbox(
    'Select Patient',
    tuple(os.listdir("./data"))
)

# select patient
result_name = st.sidebar.selectbox(
    'Result',
    tuple(os.listdir("./outputs"))
)

# set path
input_image = os.path.join(
    "./",
    "data/" + patient_name + "/" + patient_name + "_" + pattern_name + ".nii.gz",
)
output_image = os.path.join(
    "./",
    "outputs/" + "out" + patient_name + ".nii.gz",
)
model = "./saved_models/" + model_name


# set tabs
tab1, tab2 = st.tabs(["Original", "Result"])


with tab1:
    clicked1 = st.sidebar.button('upload your data')
    if clicked1:
        st.file_uploader("choose your flie")
    num = st.slider("num", 1, 155)
    utils.show_image(input_image, num)

with tab2:
    clicked = st.sidebar.button('Analysis')
    num_1 = st.slider("num2", 1, 155)
    try:
        utils.show_result(input_image, output_image, num_1)
    except RuntimeError:
        st.warning('No exit result!  Please analysis first!')
    if clicked:
        model = style2.load_model(model)
        style2.segmentation(model, patient_name)
