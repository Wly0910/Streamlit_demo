import os
import shutil
import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib
from monai.inferers import sliding_window_inference
from monai import transforms
from monai.config import print_config
from monai.metrics import DiceMetric
from monai.utils.enums import MetricReduction
from monai.networks.nets import SwinUNETR
from monai import data
from monai.data import decollate_batch
from functools import partial

import torch


def get_testloader(patient_name):
    test_files = [
        {
            "image": [
                os.path.join(
                    "./",
                    "data/" + patient_name + "/" + patient_name + "_flair.nii.gz",
                ),
                os.path.join(
                    "./",
                    "data/" + patient_name + "/" + patient_name + "_t1ce.nii.gz",
                ),
                os.path.join(
                    "./",
                    "data/" + patient_name + "/" + patient_name + "_t1.nii.gz",
                ),
                os.path.join(
                    "./",
                    "data/" + patient_name + "/" + patient_name + "_t2.nii.gz",
                ),
            ],

        }
    ]
    test_transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image"]),
            transforms.NormalizeIntensityd(
                keys="image", nonzero=True, channel_wise=True),
        ]
    )

    test_ds = data.Dataset(data=test_files, transform=test_transform)

    test_loader = data.DataLoader(
        test_ds,
        batch_size=1,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
    )
    return test_loader
