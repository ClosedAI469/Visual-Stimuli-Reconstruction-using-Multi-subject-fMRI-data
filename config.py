import torch

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)

ROI_PATH_2_MAT = "data/DS105-objectviewing-ROI-std4mm-DLY-TempAlign.mat"
WB_PATH_2_MAT = "data/DS105-objectviewing-WB-std4mm-DLY-TempAlign.mat"
# stimuli_PATH_2_MAT = "data/DS105-objectviewing-stimuli.mat"
