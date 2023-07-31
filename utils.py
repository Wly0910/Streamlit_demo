
from PIL import Image
import streamlit as st
import matplotlib.pyplot as plt
import nibabel as nib
import io
import os
import SimpleITK as stik
import numpy as np


def show_image(input_image, num):
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
    buf.close()


def show_result(input_image, output_image, num):
    img = stik.ReadImage(input_image)
    img_255 = stik.Cast(stik.RescaleIntensity(img), stik.sitkUInt8)
    seg = stik.ReadImage(output_image)
    seg = stik.LabelOverlay(img_255, seg)
    result = stik.GetArrayFromImage(seg)
    result = result[num, :, :]
    result = np.rot90(result, k=1)
    # img = nib.load(output_image).get_fdata()
    fig, ax = plt.subplots()
    ax.imshow(result)
    ax.axis('off')
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    buf.seek(1)
    img = Image.open(buf)
    st.image(img, width=512)
    buf.close()
