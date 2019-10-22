# MicroscopyGAN
Multi-defect microscopy image restoration under limited data conditions

This is Keras implementation of the paper "Multi-defect microscopy image restoration under limited data conditions" (A Razdaibiedina, J Velayutham, M Modi). Rated in Top-15 NeurIPS Medical Imaging workshop papers.

Restoration results of several microscopy defect types is shown below: <br/>
<img src="animations/example1.gif" width="120px"/> <img src="animations/example2.gif" width="120px"/> <img src="animations/example3.gif" width="120px"/> <img src="animations/example4.gif" width="120px"/> <img src="animations/example5.gif" width="120px"/> 

## Architecture 
Our pipeline consists of two GANs:
1. unpaired CIN-GAN that learns to **generate defects from limited data**
<img src="illustrations/pipeline1.png" width="600px"/> <br/>
2. paired cGAN that **restores images with multiple defects**. This GAN is trained on paired dataset that was augmented by CIN-GAN from the previous step
<img src="illustrations/pipeline2.png" width="600px"/>

### CIN-GAN
This part is inspired by <a href='https://arxiv.org/abs/1610.07629'>Conditional Instance Normalization for style transfer</a> proposed by Dumoulin et al. All layers of this GAN are shared for different defect types except for CIN layers, which are turned on / off depending on which type of defect is being synthesized. CIN layer is defined as following: <br/>
<img src="illustrations/cin.png" width="160px"/>

where i is the number of task, μ and σ are mean and the standard deviation of the input x.

We apply condition on the instance normalization for each defect type. GAN with CIN layers is trained to perform data augmentation using limited amount of unpaired ground-truth and defected images.

### cGAN
After the dataset is augmented by CIN-GAN, a conditional GAN is trained on paired high-resolution ground-truth images and defective images. The resulting cGAN is used to restore multiple types of microscopy defects. As a loss function for cGAN we used a combination of adversarial and content losses, where content loss measures image consistency in feature space of VGG16 model:<br/>
<img src="illustrations/loss.png" width="200px"/>

## Microscopy defect types
In this work we focus on three common tasks in microscopy image restoration: 
* denoising 
* axial inpainting 
* deep-learning-enabled super-resolution


## How to run
