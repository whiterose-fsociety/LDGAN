# Learning Degradation Using Generative Adversarial Networks for Image Super-Resolution


```python
conda create --name ldgan python=3.9
conda activate ldgan
cd ../
python3 -m pip install LDGAN
cd src
python3 -m pip install -r requirements
```
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
