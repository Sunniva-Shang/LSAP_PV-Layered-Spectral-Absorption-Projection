# LSAP_PV-Layered-Spectral-Absorption-Projection
*LSAP-PV: High-Fidelity Palm Vein Image Synthesis via Layered Spectral Absorption Projection-Guided Diffusion Model*
| [Paper](https://ojs.aaai.org/index.php/AAAI/article/view/37837)

## Prerequisites
- Python 3
- NVIDIA GPU + CUDA CuDNN

# Getting Started

## Installation
- Clone this repo:
```bash
git clone https://github.com/Sunniva-Shang/LSAP_PV-Layered-Spectral-Absorption-Projection
``` 
- Install dependencies:
`
pip install -r requirements.txt
`

## Palm Vein Images Generation

### Palm Vascular Tree Generation
- `cd pvtree;
  bash get_patterns.sh`
- The output are saved in `./pvtree/pv_pattern_results/palmvein`

### Images Generation
- Download [DM-ckp](https://drive.google.com/file/d/1K85P8HTwGZ99v5jTL9dfOtZzlwZ6kBE8/view?usp=share_link), unzip it and place it in `./checkpoints`.
- `bash sample_muti_gpus.sh`
- The generated images are saved in `./results`.

## Train the Diffusion Model
### Preparing Training Data
- Download [CycleGAN-ckp](https://drive.google.com/file/d/1O9HVMWcOjLyVW26cvkO0OYdisJYYZW7d/view?usp=share_link), unzip it and place it in `./cyclegan/checkpoints`.
- Extracting palm vein patterns from real data with a pre-trained model `bash test.sh`
- The generated images are saved in `./cyclegan/results`.

### Storage Format of Training Data
<pre>
traindata             
├── cond                 
│   ├── {DatasetName}_{idname}  # DatasetName={casia/HFUT/polyu/TongJi}         
│   │   ├── img1.png/jpg/...   
│   │   └── ...   
│   └── ...              
└── vein                
    ├── {DatasetName}_{idname}             
    │   ├── img1.png/jpg/...
    │   └── ...          
    └── ... 
</pre>              

### Start Training Diffusion Model
- `bash train.sh`
- More details see paper [IDDPM](https://ojs.aaai.org/index.php/AAAI/article/view/28039).


### Citation
If you find this useful for your research, please use the following.

```
@inproceedings{shang2025pvtree,
  title={PVTree: Realistic and controllable palm vein generation for recognition tasks},
  author={Shang, Sheng and Zhao, Chenglong and Zhang, Ruixin and Jin, Jianlong and Zhang, Jingyun and Guo, Rizen and Ding, Shouhong and Wu, Yunsheng and Zhao, Yang and Jia, Wei},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={39},
  number={7},
  pages={6767--6775},
  year={2025}
}

@inproceedings{nichol2021improved,
  title={Improved denoising diffusion probabilistic models},
  author={Nichol, Alexander Quinn and Dhariwal, Prafulla},
  booktitle={International conference on machine learning},
  pages={8162--8171},
  year={2021},
  organization={PMLR}
}

```

If you have any questions or encounter any issues with the code, please feel free to contact me (email: shengshang@mail.hfut.edu.cn). 
I would be more than happy to assist you in any way I can.

### Acknowledgements
This code borrows heavily from the [PVTree](https://github.com/Sunniva-Shang/PVTree-palmvein-generation) and
[IDDPM](https://github.com/openai/improved-diffusion).
