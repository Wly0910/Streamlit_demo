
import nibabel as nib
import numpy as np
import torch
import torch.onnx
from monai.networks.nets import SwinUNETR
from functools import partial
import utils
import streamlit as st
from monai.inferers import sliding_window_inference
import testdata

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@st.cache_resource
def load_model(model_path):
    print('load model')
    with torch.no_grad():
        model = SwinUNETR(
            img_size=(128, 128, 128),
            in_channels=4,
            out_channels=3,
            feature_size=48,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            dropout_path_rate=0.0,
            use_checkpoint=True,
        ).to(device)
        model.load_state_dict(torch.load(model_path)["state_dict"])
        model.to(device)
        model.eval()
        model_inferer_test = partial(
            sliding_window_inference,
            roi_size=[128, 128, 128],
            sw_batch_size=1,
            predictor=model,
            overlap=0.6,
        )
        return model_inferer_test


@st.cache_data
def segmentation(_model, patient_name):

    with torch.no_grad():
        for batch_data in testdata.get_testloader(patient_name):
            image = batch_data["image"].cuda()
            affine = batch_data["image_meta_dict"]["original_affine"][0].numpy()

            prob = torch.sigmoid(_model(image))
            seg = prob[0].detach().cpu().numpy()
            seg = (seg > 0.5).astype(np.int8)
            seg_out = np.zeros((seg.shape[1], seg.shape[2], seg.shape[3]))
            seg_out[seg[1] == 1] = 2
            seg_out[seg[0] == 1] = 1
            seg_out[seg[2] == 1] = 4
            nib.save(nib.Nifti1Image(seg_out.astype(
                np.uint8), affine), "outputs/out" + patient_name + ".nii.gz")
