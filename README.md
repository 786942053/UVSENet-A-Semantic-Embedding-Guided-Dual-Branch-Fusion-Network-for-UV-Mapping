# UVSENet: A Semantic-Embedded Dual Branch Fusion Network for Urban Villages Mapping

> Official implementation of “UVSENet: A Semantic-Embedded dual branch fusion network for urban villages mapping”
> [International Journal of Applied Earth Observation and Geoinformation], [2026]
[![Paper]([![Paper](https://img.shields.io/badge/Paper-PDF-red)]([[Paper Link](https://authors.elsevier.com/sd/article/S1569-8432(26)00353-5)]))]

<img width="554" height="382" alt="image" src="https://github.com/user-attachments/assets/d5407065-c2d5-4caa-8efb-6f42a36873cc" />



## Environment Setup

### Requirements

* Python: `[> 3.8]`
* PyTorch: `[> 1.8.1]`
* CUDA: `[> 11.1]`
* Operating System: `[Linux]`

### Installation

```bash
cd UVSENet_mian
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Dataset Preparation

### Dataset Download

Download the dataset from:

* Dataset1: obtained from Ref. (https://doi.org/10.1016/j.jag.2025.104631)
* Download link: [[Dataset link](https://pan.baidu.com/s/1CZ9vQB22jRQSlaGxGRkpHg?pwd=rxw9)]

* Dataset2: Custom dataset
* Download link: [[Dataset link](https://pan.baidu.com/s/1D1LuZQiZSGipq9v7njAppA?pwd=8djc)]

### Dataset Structure

Organize the dataset as follows:

```text
data/
├── train/
│   ├── images/
│   └── masks/
├── val/
│   ├── images/
│   └── masks/
└── test/
    └── images/
    └── masks/
embed
├── train/
│   └── images/
├── val/
│   └── images/
└── test/
    └── images/
```


## Pretrained Models

Download pretrained weights from UVSENet:

* [https://pan.baidu.com/s/1CZ9vQB22jRQSlaGxGRkpHg?pwd=rxw9]


---

## Training

Train the model using the default configuration:

```bash
CUDA_VISIBLE_DEVICES=0 python train.py -c config/UVData/UVSENet.py
```


---

## Evaluation

Evaluate a trained model:

```bash
CUDA_VISIBLE_DEVICES=0 python test.py -c config/UVData/UVSENet.py -o test_results/UVData/UVSENet/ --rgb
```

Evaluation results will be saved in:

```text
test_results/
└── UVData/
    └── UVSENet/
```

---

## Inference

Run inference on a single image:

```bash
python inference.py \
    --input [path_to_image] \
    --checkpoint checkpoints/[checkpoint_name].pth \
    --output outputs/inference_result.png
```

Run inference on a folder:

```bash
python inference.py \
    --input_dir [path_to_image_folder] \
    --checkpoint checkpoints/[checkpoint_name].pth \
    --output_dir outputs/inference_results
```

Example:

```bash
python inference.py \
    --input demo/example.jpg \
    --checkpoint checkpoints/[checkpoint_name].pth \
    --output outputs/example_prediction.png
```


## Acknowledgements

This project is built upon or inspired by:

* Repository1: https://github.com/Henryjiepanli/Uncertainty-aware-Network
* paper 1 and data 1: https://doi.org/10.1016/j.jag.2025.104631
  
## If you use our code, models, or dataset processing pipeline in your research, please cite our paper:

@article{[citation_key],
  title     = {UVSENet: A Semantic-Embedded dual branch fusion network for urban villages mapping},
  author    = {Shaobo Qiu and Shi Shen and Changqing Song and Cansong Li},
  journal   = {International Journal of Applied Earth Observation and Geoinformation},
  year      = {2026},
  volume    = {152},
  article   = {105437},
  doi       = {10.1016/j.jag.2026.105437}
}
## Citation

If you use our code, models, or dataset processing pipeline in your research, please cite our paper:

```bibtex
@article{UVSENet,
  title     = {UVSENet: A Semantic-Embedded dual branch fusion network for urban villages mapping},
  author    = {Shaobo Qiu and Shi Shen and Changqing Song and Cansong Li},
  journal   = {International Journal of Applied Earth Observation and Geoinformation},
  year      = {2026},
  volume    = {152},
  article   = {105437},
  doi       = {10.1016/j.jag.2026.105437}
}
```
