import torch

# PyTorch device
DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)

# data paths for Tony's data
ROI_PATH_2_MAT = "data/DS105-objectviewing-ROI-std4mm-DLY-TempAlign.mat"
WB_PATH_2_MAT = "data/DS105-objectviewing-WB-std4mm-DLY-TempAlign.mat"

# data constants for Tony's data
N_SUBJECTS = 6
N_RUNS_PER_SUBJECT = 12
N_RUNS_TOTAL = N_SUBJECTS * N_RUNS_PER_SUBJECT
N_SCANS_PER_RUN = 121
N_VOXELS_ROI = 2294
N_VOXELS_WB = 19174
N_SCANS_PER_SUBJECT = N_RUNS_PER_SUBJECT * N_SCANS_PER_RUN
N_SCANS_TOTAL = N_SUBJECTS * N_SCANS_PER_SUBJECT
N_CATEGORIES_IMAGES = 8
