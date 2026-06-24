# UVSENet: A Semantic-Embedded Dual Branch Fusion Network for Urban Villages Mapping
The published dataset used in this study was obtained from Ref. (https://doi.org/10.1016/j.jag.2025.104631), and the trained weight files corresponding to models trained on the custom dataset and the published dataset, respectively, are available for download at (https://pan.baidu.com/s/1CZ9vQB22jRQSlaGxGRkpHg?pwd=rxw9)
We are pleased to announce that our paper has been accepted for publication in The International Journal of Applied Earth Observation and Geoinformation. The complete code has been uploaded.
The annotated UV dataset can be downloaded from [...](https://pan.baidu.com/s/1D1LuZQiZSGipq9v7njAppA?pwd=8djc)

> Official implementation of “UVSENet: A Semantic-Embedded dual branch fusion network for urban villages mapping”
> [International Journal of Applied Earth Observation and Geoinformation], [2026]
[![Paper]([![Paper](https://img.shields.io/badge/Paper-PDF-red)]([[Paper Link](https://authors.elsevier.com/sd/article/S1569-8432(26)00353-5)]))]




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

---

## Reproducing Results

The following table reports the main results from the paper.

| Method   |   Dataset | Metric 1 | Metric 2 | Metric 3 |
| -------- | --------: | -------: | -------: | -------: |
| Baseline | [Dataset] |  [Value] |  [Value] |  [Value] |
| Ours     | [Dataset] |  [Value] |  [Value] |  [Value] |

To reproduce the reported result:

```bash
python test.py \
    --config configs/[paper_setting].yaml \
    --checkpoint checkpoints/[paper_checkpoint].pth
```

Note that minor differences may occur because of:

* Random seeds;
* GPU hardware;
* CUDA and PyTorch versions;
* Dataset preprocessing;
* Floating-point computation differences.

---

## Troubleshooting

### 1. `ModuleNotFoundError`

Install missing dependencies:

```bash
pip install [package_name]
```

Or reinstall all required packages:

```bash
pip install -r requirements.txt
```

### 2. CUDA Out of Memory

Reduce the batch size:

```bash
python train.py --batch_size 2
```

You may also reduce image size or use mixed-precision training.

### 3. Dataset Path Error

Check the dataset path in:

```text
configs/[config_name].yaml
```

Ensure that the folder structure matches the format described in the **Dataset Preparation** section.

### 4. Model Checkpoint Cannot Be Loaded

Check that:

* The checkpoint path is correct;
* The model configuration matches the checkpoint;
* PyTorch and CUDA versions are compatible;
* The checkpoint file is fully downloaded.

---

## Citation

Please cite our paper if you find this repository useful:

```bibtex
@article{[citation_key],
  title     = {[Paper Title]},
  author    = {[Author 1] and [Author 2] and [Author 3]},
  journal   = {[Journal Name]},
  year      = {[Year]},
  volume    = {[Volume]},
  number    = {[Issue]},
  pages     = {[Pages]},
  doi       = {[DOI]}
}
```

---

## License

This project is released under the [License Name] License. See the [LICENSE](LICENSE) file for details.

---

## Contact

For questions related to the code or paper, please contact:

* Name: [Your Name]
* Email: [Your Email]
* GitHub: [Your GitHub Profile]

---

## Acknowledgements

This project is built upon or inspired by:

* [Repository / paper 1]
* [Repository / paper 2]
* [Dataset provider]
* [Funding source, if applicable]

