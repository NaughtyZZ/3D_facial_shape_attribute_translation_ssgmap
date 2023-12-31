# Unpaired Multi-domain Attribute Translation of 3D Facial Shapes with a Square and Symmetric Geometric Map

by Zhenfeng Fan, Zhiheng Zhang, Shuang Yang, Chongyang Zhong, Min Cao and Shihong Xia

## Introduction

This repository is built for the official implementation of the paper "Unpaired Multi-domain Attribute Translation of 3D Facial Shapes with a Square and Symmetric Geometric Map".
To view the full paper, please visit the online version at [https://arxiv.org/abs/2308.13245](https://arxiv.org/abs/2308.13245).

## Abstract
While impressive progress has recently been made in image-oriented facial attribute translation, shape-oriented 3D facial attribute translation remains an unsolved issue. This is primarily limited by the lack of 3D generative models and ineffective usage of 3D facial data. We propose a learning framework for 3D facial attribute translation to relieve these limitations. Firstly, we customize a novel geometric map for 3D shape representation and embed it in an end-to-end generative adversarial network. The geometric map represents 3D shapes symmetrically on a square image grid, while preserving the neighboring relationship of 3D vertices in a local least-square sense. This enables effective learning for the latent representation of data with different attributes. Secondly, we employ a unified and unpaired learning framework for multi-domain attribute translation. It not only makes effective usage of data correlation from multiple domains, but also mitigates the constraint for hardly accessible paired data. Finally, we propose a hierarchical architecture for the discriminator to guarantee robust results against both global and local artifacts. We conduct extensive experiments to demonstrate the advantage of the proposed framework over the state-of-the-art in generating high-fidelity facial shapes. Given an input 3D facial shape, the proposed framework is able to synthesize novel shapes of different attributes, which covers some downstream applications, such as expression transfer, gender translation, and aging.
## Usage

### Training
We provide the training code for the model. Since we are not authorized for the facescape dataset, we suggest that the users can download it themselves at [FaceScape Project](https://github.com/zhuhao-nju/facescape) and then formalize the data with our provided gmap at  `` .\template\ `` referring to our paper. Then, the user may run the training code: 

```
python main.py
```

### Testing
We provide a testing demo as well as a trained model. The results in *.ply* file format can be viewed by the opensource [meshlab software](https://www.meshlab.net/)  at the folder `` .\test_data\ ``. The user can run by

```
python demo.py
```
A raw scaning face requires to be registed to the provided template first before applying the trained model to it. The user may refer to our another publication [*"Towards Fine-Grained Optimal 3D Face Dense Registration: An Iterative Dividing and Diffusing Method (IJCV2023)"*](https://doi.org/10.1007/s11263-023-01825-7) for the registration. All the codes for registration are at [https://github.com/NaughtyZZ/3D_face_dense_registration](https://github.com/NaughtyZZ/3D_face_dense_registration).
## Remarks

- [x] The Bi-directional sampling process between a 3D facial shape and its representation on a Gmap is shown as follows. 
<img src="figures\Bi-direction sampling.png" alt="show" style="zoom: 67%;" />

- [x] The 3D template and the gmap is placed at the fold  `` .\template\ ``.

- [x] Some examples for attribute translation of 3D facial shape is shown as follows.
<img src="figures\figure_show.png" alt="show" style="zoom: 67%;" />
<img src="figures\figure_multi_domain_a.png" alt="multi_domain_a" style="zoom: 67%;" />
<img src="figures\gender_age_sup.png" alt="gender_age_sup" style="zoom: 67%;" />

## Dependencies and Requirements

- [x] Python 3.8.17
- [x] PyTorch  2.0.1 && torchvision 0.15.2
- [x] scipy 
- [x] h5py
- [x] tensorboard and tensorboardx

## Acknowledgement

Our code is partially borrowed from [StarGAN](https://github.com/yunjey/stargan).

Our training code requires the [Facescape dataset](https://github.com/zhuhao-nju/facescape).

## Sponsorships

This work is supported in part by the National Key Research and Development Program of China (No. 2022YFF0902302), the National Science Foundation of China (No. 62106250 and No. 62002252), and China Postdoctoral Science Foundation (No. 2021M703272).

## Bibtex
If you find this project helpful to your research, please consider citing:

```
@article{fan2023unpaired,
  title={Unpaired Multi-domain Attribute Translation of 3D Facial Shapes with a Square and Symmetric Geometric Map},
  author={Fan, Zhenfeng and Zhang, Zhiheng and Yang, Shuang and Zhong, Chongyang and Cao, Min and Xia, Shihong},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={20828--20838},
  year={2023}
}
```
