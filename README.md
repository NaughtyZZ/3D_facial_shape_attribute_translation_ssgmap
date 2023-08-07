# Unpaired Multi-domain Attribute Translation of 3D Facial Shapes with a Square and Symmetric Geometric Map (ICCV2023)

by Zhenfeng Fan, Zhiheng Zhang, Shuang Yang, Chongyang Zhong and Shihong Xia

## Introduction

This repository is built for the official implementation of the paper "Unpaired Multi-domain Attribute Translation of 3D Facial Shapes with a Square and Symmetric Geometric Map" published at *International Conference on Computer Vision* in October, 2023.
To view the full version of this paper, please visit the website, which provides the online version.

## Abstract
While impressive progress has recently been made in image-oriented facial attribute translation, shape-oriented 3D facial attribute translation remains an unsolved issue. This is primarily limited by the lack of 3D generative models and ineffective usage of 3D facial data. We propose a learning framework for 3D facial attribute translation to relieve these limitations. Firstly, we customize a novel geometric map for 3D shape representation and embed it in an end-to-end generative adversarial network. The geometric map represents 3D shapes symmetrically on a square image grid, while preserving the neighboring relationship of 3D vertices in a local least-square sense. This enables effective learning for the latent representation of data with different attributes. Secondly, we employ a unified and unpaired learning framework for multi-domain attribute translation. It not only makes effective usage of data correlation from multiple domains, but also mitigates the constraint for hardly accessible paired data. Finally, we propose a hierarchical architecture for the discriminator to guarantee robust results against both global and local artifacts. We conduct extensive experiments to demonstrate the advantage of the proposed framework over the state-of-the-art in generating high-fidelity facial shapes. Given an input 3D facial shape, the proposed framework is able to synthesize novel shapes of different attributes, which covers some downstream applications, such as expression transfer, gender translation, and aging.
## Usage

### Training
  

```
  
```
### Testing
  
```
  
```

#### Dependencies and Requirements

- [x] Python 3.8.17
- [x] PyTorch  2.0.1 && torchvision 0.15.2
- [x] scipy 
- [x] h5py
- [x] tensorboard and tensorboardx


## Sponsorships



## Bibtex
If you find this project helpful to your research, please consider citing:

```
@article{fan2023towards,
  title={Towards Fine-Grained Optimal 3D Face Dense Registration: An Iterative Dividing and Diffusing Method},
  author={Fan, Zhenfeng and Peng, Silong and Xia, Shihong},
  journal={International Journal of Computer Vision},
  pages={1--21},
  month = {June},
  year={2023},
  publisher={Springer}
}
```
