# AB-BNN
This is the pytorch implementation of our paper **A&B BNN: Add&Bit-Operation-Only Hardware-Friendly Binary Neural Network** published in CVPR 2024.

This work proposes a novel hardware-friendly binary neural network architecture that does not require any multiplication operations.

## Abstract
Binary neural networks utilize 1-bit quantized weights and activations to reduce both the model’s storage demands and computational burden.
However, advanced binary architectures still incorporate millions of inefficient and nonhardware-friendly full-precision multiplication operations.
A&B BNN is proposed to directly remove part of the multiplication operations in a traditional BNN and replace the rest with an equal number of bit operations, introducing the mask layer and the quantized RPReLU structure based on the normalizer-free network architecture.
The mask layer can be removed during inference by leveraging the intrinsic characteristics of BNN with straightforward mathematical transformations to avoid the associated multiplication operations.
The quantized RPReLU structure enables more efficient bit operations by constraining its slope to be integer powers of 2.
Experimental results achieved 92.30%, 69.35%, and 66.89% on the CIFAR-10, CIFAR-100, and ImageNet datasets, respectively, which are competitive with the state-of-the-art.
Ablation studies have verified the efficacy of the quantized RPReLU structure, leading to a 1.14% enhancement on the ImageNet compared to using a fixed slope RLeakyReLU.
The proposed add&bit-operation-only BNN offers an innovative approach for hardware-friendly network architecture.

## Citation
If you find our code useful for your research, please consider citing:

## Requirements
- python3, pytorch2.0.1