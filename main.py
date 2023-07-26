import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image
import os
import style2
import nibabel as nib
import io

st.title('PyTorch Style Transfer')

pattern_name = st.sidebar.selectbox(
    'Select Your MRI',
    ('t1', 't1ce', 'flair', 't2')
)

style_name = st.sidebar.selectbox(
    'Select Model',
    tuple(os.listdir("./saved_models"))
)

patient_name = st.sidebar.selectbox(
    'Select Style',
    tuple(os.listdir("./data"))
)

table1, table2 = st.tabs(["old", "new"])

model = "./saved_models/" + style_name

input_image = os.path.join(
    "./",
    "data/" + patient_name + "/" + patient_name + "_" + pattern_name + ".nii.gz",
)

output_image = os.path.join(
    "./",
    "outputs/" + "out" + patient_name + ".nii.gz",
)

with table1:
    clicked1 = st.sidebar.button('Show your data')
    num = st.slider("num", 1, 155)
    if clicked1:
        # st.write('### Your orginal image')
        img = nib.load(input_image).get_fdata()
        fig, ax = plt.subplots()
        ax.imshow(img[:, :, num - 1], cmap='gray')
        ax.axis('off')

        # 将matplotlib figure对象转换为PIL Image对象
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
        buf.seek(0)
        img = Image.open(buf)
        st.image(img, width=512)
        # buf.close()

with table2:
    clicked = st.button('Analysis')

    if clicked:
        model = style2.load_model(model)
        style2.segmentation(model, patient_name)
        # st.write('### Your Result')
        img = nib.load(output_image).get_fdata()
        fig, ax = plt.subplots()
        ax.imshow(img[:, :, 78], cmap='gray')
        ax.axis('off')
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(1)
        img = Image.open(buf)
        st.image(img, width=1024)
        buf.close()
