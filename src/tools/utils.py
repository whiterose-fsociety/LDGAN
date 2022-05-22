import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
import torchvision
import os
import config.config as config
import numpy as np
from PIL import Image
from torchvision.utils import save_image
import matplotlib.pyplot as plt


experiment_number = config.experiment_facts['experiment_number']  # 1
run_number = config.experiment_facts['run_number']  # 1


def gradient_penalty(critic, real, fake, device='cpu'):
    BATCH_SIZE, C, H, W = real.shape
    epsilon = torch.rand((BATCH_SIZE, 1, 1, 1)).repeat(1, C, H, W).to(device)
    interpolated_images = real*epsilon + fake * \
        (1-epsilon)  # 90% real image + 10% fake

    # Calculate Critic Score
    mixed_scores = critic(interpolated_images)

    gradient = torch.autograd.grad(
        inputs=interpolated_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True
    )[0]

    gradient = gradient.view(gradient.shape[0], -1)
    gradient_norm = gradient.norm(2, dim=1)  # L2 norm
    gradient_penalty = torch.mean((gradient_norm-1)**2)
    return gradient_penalty


def imshow(img, unnormalize=False):
    if unnormalize:
        img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def get_crop(lr_image, hr_image):
    i, j, h, w = torchvision.transforms.RandomCrop.get_params(hr_image, output_size=(
        config.high_res, config.high_res))  # Get random crop of tensor hr image
    hr_pil = torchvision.transforms.ToPILImage()(
        hr_image)  # Pil Version of hr_image
    lr_pil = torchvision.transforms.ToPILImage()(
        lr_image)  # Pil Version of lr_image
    hcrop = TF.crop(hr_pil, i, j, h, w)  # Get 128x128 crop of hr image
    # Get 128x128 crop of lr image
    lcrop = TF.crop(lr_pil, i//4, j//4, h//4, w//4)
    hcrop_tensor = config.test_transform(image=np.asarray(hcrop))[
        'image']  # Tensor version of 128 crophcrop_tensor
    lcrop_tensor = config.test_transform(image=np.asarray(lcrop))[
        'image']  # Tensor version of 128 crop
    return lcrop_tensor, hcrop_tensor


def plot_examples(l2h, h2l, epoch=None, idx=None, verbose=True, unnormalize=False, split=True):
    if type(epoch) is None and type(idx) is None:
        lr_filename = config.evaluation_lr_evaluation_dataset_folder
        hr_filename = config.evaluation_hr_evaluation_dataset_folder

        lr_files = sorted(os.listdir(lr_filename))
        hr_files = sorted(os.listdir(hr_filename))
    else:
        lr_filename = config.evaluation_lr_during_training_dataset_folder
        hr_filename = config.evaluation_hr_during_training_dataset_folder

        lr_files = sorted(os.listdir(lr_filename))
        hr_files = sorted(os.listdir(hr_filename))
    l2h.eval()
    h2l.eval()
    for lr_file, hr_file in zip(lr_files, hr_files):
        lr_image_name = lr_filename + lr_file
        hr_image_name = hr_filename + hr_file

        lr_image = Image.open(lr_image_name)
        hr_image = Image.open(hr_image_name)

        with torch.no_grad():
            lr_image = config.test_transform(image=np.asarray(lr_image))[
                'image'].unsqueeze(0).to(config.DEVICE)
            hr_image = config.test_transform(image=np.asarray(hr_image))[
                'image'].unsqueeze(0).to(config.DEVICE)
            zs = np.random.randn(
                config.training_facts['batch_size'], 1, config.high_res).astype(np.float32)
            zs = torch.from_numpy(zs).to(config.DEVICE)

            lcrop, hcrop = get_crop(lr_image, hr_image)
            lcrop = lcrop.unsqueeze(0)
            hcrop = hcrop.unsqueeze(0)  # [1,c,h,w]

            # Downsample the original hr image - Want to see how well the Learned Degradation Performs
            dowsampled_image = h2l(hcrop, zs)

            # Upsample the original lr image - Want to see how the model upsamples
            upsampled_image = l2h(lr_image)

            if unnormalize:
                dowsampled_image = dowsampled_image*0.5 + 0.5
                upsampled_image = upsampled_image*0.5 + 0.5

            downsampled_image = torch.cat([downsampled_image, lcrop], dim=0)
            # [Original Low-Resolution, Generated Low-Resolution]
            downsampled_image = torchvision.utils.make_grid(downsampled_image)

            if split:
                upsampled_image = torch.cat([upsampled_image, hr_image], dim=0)
                # [Generated Low-Resolution, Original Low-Resolution]
                upsampled_image = torchvision.utils.make_grid(upsampled_image)

        if type(epoch) is None and type(idx) is None:
            save_lr_folder = config.result_lr_dir + \
                config.results_facts['evaluation']
            save_lr_file = save_lr_folder + f'{lr_file}'

            save_hr_folder = config.result_hr_dir + \
                config.results_facts['evaluation']
            save_hr_file = save_hr_folder + f'{hr_file}'
        else:
            # If the folders do not exist, make the directories
            save_lr_folder = config.result_lr_dir + \
                config.results_facts['during_training_images']
            save_lr_file = save_lr_folder + \
                f'epoch_{epoch}_idx_{idx}_{lr_file}'

            save_hr_folder = config.result_hr_dir + \
                config.results_facts['during_training_images']
            save_hr_file = save_hr_folder + \
                f'epoch_{epoch}_idx_{idx}_{hr_file}'
        if not os.path.exists(save_lr_folder):
            os.makedirs(save_lr_folder)
        if not os.path.exists(save_hr_folder):
            os.makedirs(save_hr_folder)
        if verbose:
            config.print_message(
                "Saving Image LR Images Inside '{}' ".format(save_lr_folder))
            config.print_message(
                "Saving Image HR Images Inside '{}' ".format(save_hr_folder))
        save_image(dowsampled_image, save_lr_file)
        save_image(upsampled_image, save_hr_file)
    l2h.train()
    h2l.train()
