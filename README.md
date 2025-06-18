# Codes for Multi-Modality Medical Image Fusion by Edge Supervising and Multi-Scale Attention Features Extraction


## Abstract
Since the development of deep learning, multimodal medical image fusion (MMIF) has achieved both efficiency and 
real-time performance.} However, most existing deep learning-based fusion methods primarily focus on the overall 
network architecture, often overlooking the intrinsic characteristics of the source images. From the perspectives of 
SPECT and PET imaging and the continuity of biological information, the edge regions of these functional images should 
receive greater attention during the fusion process. Furthermore, the pseudocolor should be separated before fusion and 
reintroduced afterward. Since functional images such as SPECT and PET typically suffer from low clarity, directly fusing
them without proper processing may obscure texture details in the resulting image. To address these challenges, we 
propose an end-to-end encoder–decoder network for multimodal medical image fusion, termed EFCNet. The encoder comprises 
three main components: a Smooth Edge Extraction Module (SEEM), a Multi-Scale Attention Module (MSAM), and the E-Fusion 
module. The decoder reconstructs the fused image from these features. Specifically, SEEM extracts and smooths the edge
information of functional source images (SPECT and PET), thereby mitigating the issues mentioned above. MSAM captures 
both local details and global contextual features while adaptively emphasizing more informative channels. E-Fusion then 
performs effective fusion of the extracted local and global features. Notably, our model is trained on a single dataset 
to obtain the pretrained weights, yet it achieves impressive results when tested on other datasets, demonstrating the 
strong generalization capability of the proposed method.


## Usage


### 1.Requirements
To run the models, you need to have the following libraries installed:
- python=3.10.14
- torch==2.0.0
- torchaudio==2.0.1
- torchvision==0.15.1
- matplotlib==3.9.1
- numpy==1.26.4
- opencv-python==4.10.0.84


### 2.Network Architecture
All of our network architecture code is organized within the **`networks`** folder.


### 3.Dataset
Part of our experimental dataset is sourced from the public dataset provided by Harvard Medical School: 
[https://www.med.harvard.edu/AANLIB/home.html](https://www.med.harvard.edu/AANLIB/home.html).
The GFP-PC dataset also originates from a publicly available dataset; if needed, please contact the authors for access.



### 4.Pre-trained files
The files generated after training the SEEM module are saved in the `model_smooth_edge` folder.
If you wish to retrain this module, please run the following command:
```bash
python train_ee.py
```
The pretrained model files after training the entire module are saved in the `model_save` folder.
If you wish to retrain the model, please run the following command:
```bash
python Train.py
```


### 5.Test
If you wish to test our fusion method, please ensure that the correct version of PyTorch is installed, and then run the following command:
```bash
python test.py
```

