### Hydra: Accurate Multi-Modal Leaf Wetness Sensing with mm-Wave and Camera

This repository contains the dataset and model architecture of the MobiCom'24 paper "Hydra: Accurate Multi-Modal Leaf Wetness Sensing with mm-Wave and Camera".
The dataset and architecture are still under development.

## Overview
![overview image](figure/overview.png "Title")
Hydra contains three main procedures: Single-Depth Feature Extraction, Multi-Depth Feature Detection, and classifier. We fuse the multi-modality and extract wetness features in the Single-Depth Feature Extraction phase. The Multi-Depth Feature Detection stage leverages these features at a 3D level, culminating in utilizing a leaf wetness classification algorithm to derive the final wetness assessment.


## Repository structure
This repository includes the following contents:


	|- MobiCom24-Hydra
		|- dataset       - The dataset of the empirical 
  			|- RGB Image       - The RGB Camera Image



## Reference
[1] Yimeng Liu, Maolin Gan, Huaili Zeng, Li Liu, Younsuk Dong and Zhichao Cao 2024. Hydra: Accurate Multi-Modal Leaf Wetness Sensing with mm-Wave and Camera. In Proceedings of the 30th Annual International Conference on Mobile Computing and Networking (Washington D.C. USA) (ACM MobiCom ’24). Association for Computing Machinery, New York, NY, USA, 230–245.
