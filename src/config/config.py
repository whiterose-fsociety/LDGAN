import yaml
import os
from PIL import Image
import albumentations as A
import torch
import torch.optim as optim
from albumentations.pytorch import ToTensorV2
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR
from torch import nn
from torch.utils.tensorboard import SummaryWriter


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

experiments_dir = "/home-mscluster/mmolefe/Playground/Artificial Intelligence/Research/Learning Degradation Using Generative Adversarial Networks For Image Super-Resolution/End-To-End Network/src/experiments/"
experiment_name = "exp01_run01_config.yaml"
experiment = experiments_dir + experiment_name
configuration = yaml.load(open(experiment, "r"), Loader=yaml.FullLoader)

# Experiment Facts
experiment_facts = configuration["experiment_facts"]
checkpoint_facts = experiment_facts['checkpoints']

# Dataset Facts
dataset_facts = configuration["dataset_facts"]
dataset = dataset_facts["dataset_root_dir"] + dataset_facts["dataset_name"]
lr_train_dataset_folder_name = dataset + dataset_facts["lr_train_dataset"]
lr_valid_dataset_folder_name = dataset + dataset_facts["lr_valid_dataset"]
lr_test_dataset_folder_name = dataset + dataset_facts["lr_test_dataset"]
hr_train_dataset_folder_name = dataset + dataset_facts["hr_train_dataset"]
hr_valid_dataset_folder_name = dataset + dataset_facts["hr_valid_dataset"]
hr_test_dataset_folder_name = dataset + dataset_facts["hr_test_dataset"]
resolution_facts = dataset_facts["resolution"]
low_res, high_res = resolution_facts["low_res"], resolution_facts["high_res"]


# Transform facts
transform_facts = dataset_facts["transforms"]
mean = transform_facts["mean"]
std = transform_facts["std"]
highres_transform = A.Compose([A.Normalize(mean=mean, std=std), ToTensorV2()])

lowres_transform = A.Compose(
    [
        A.Resize(width=low_res, height=low_res, interpolation=Image.BICUBIC),
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ]
)

hr_crop_transforms = A.Compose(
    [
        A.RandomCrop(width=high_res, height=high_res),
        A.HorizontalFlip(p=dataset_facts["transforms"]["horizontal"]["p"]),
        A.RandomRotate90(p=dataset_facts["transforms"]["random_rotate"]["p"]),
    ]
)

test_transform = A.Compose(
    [
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ]
)


# Model Facts
model_facts = configuration["model_facts"]

# Training Facts
training_facts = configuration["training_facts"]
upsampling_facts = training_facts['upsampling']
downsampling_facts = training_facts['downsampling']

# Loss Facts
loss_facts = configuration["loss_facts"]

# Evaluation Facts
evaluation_facts = configuration["evaluation_facts"]
evaluation_dir = evaluation_facts['root_dir'] + dataset_facts['dataset_name']
evaluation_lr_during_training_dataset_folder = evaluation_dir + evaluation_facts['lr_during_training_images']
evaluation_hr_during_training_dataset_folder = evaluation_dir + evaluation_facts['hr_during_training_images']

evaluation_lr_evaluation_dataset_folder = evaluation_dir + evaluation_facts['lr_evaluation_images']
evaluation_hr_evaluation_dataset_folder = evaluation_dir + evaluation_facts['hr_evaluation_images']

# Results Facts
results_facts = configuration["results_facts"]
result_dir = results_facts['root_dir'] + \
    f"{experiment_facts['experiment_name']}/" + \
    f"{results_facts['model_name']}/"
result_lr_dir = result_dir + results_facts['lr_results']
result_hr_dir = result_dir + results_facts['hr_results']


def print_message(message):
    print(f"================> {message} <================")
