# Learning Degradation Using Generative Adversarial Networks for Image Super-Resolution

## Experiment 1: Vanilla End-To-End Network

* Run 1
  * Upsample: ESRGAN
    * Adversarial Network
  * Downsample: Vanilla GAN
    * Standard "To Learn Degradation Loss"

* Run 2
  * Upsample: ESRGAN
    * Adversarial Network
  * Downsample: Vanilla GAN
  * Charbonner Loss

* Run 3
  * Upsample: ESRGAN
    * Adversarial Network
  * Downsample: Vanilla GAN
    * Texture Loss

## Experiment 2: Denoising Encoder End-To-End Network

* Run 1
  * Upsample: ESRGAN
    * Adversarial Network
  * Denoiser: XDCNN
    * Texture Loss
  * Downsample: Vanilla GAN
    * Standard "To Learn Degradation Loss"

* Run 2
  * Upsample: ESRGAN
    * Adversarial Network
  * Denoiser: XDCNN
    * Charbonner Loss
  * Downsample: Vanilla GAN
    * Standard "To Learn Degradation Loss"

* Run 3
  * Upsample: ESRGAN
    * Adversarial Network
  * Denoiser: xDCNN
    * Texture Loss
  * Downsample: Vanilla GAN
    * Standard "To Learn Degradation Loss"

* Run 4
  * Upsample: ESRGAN
    * Adversarial Network
  * Denoiser: Ridnet
    * Charbonner Loss
  * Downsample: Vanilla GAN
    * Standard "To Learn Degradation Loss"

## Experiment 3: Skip-Connection Downsample GAN:  End-To-End Network

* Run 1
  * Upsample: ESRGAN
    * Adversarial Network
  * Downsample: Skip-Connection GAN
    * Standard "To Learn Degradation Loss"

* Run 2
  * Upsample: ESRGAN
    * Adversarial Network
  * Downsample: Skip-Connection GAN
    * Charbonner Loss

* Run 3
  * Upsample: ESRGAN
    * Adversarial Network
  * Downsample: Skip-Connection GAN
    * Texture Loss
