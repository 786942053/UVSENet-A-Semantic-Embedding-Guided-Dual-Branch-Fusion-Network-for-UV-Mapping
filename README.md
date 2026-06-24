# UVSENet: A Semantic-Embedded Dual Branch Fusion Network for Urban Villages Mapping
The published dataset used in this study was obtained from Ref. (https://doi.org/10.1016/j.jag.2025.104631), and the trained weight files corresponding to models trained on the custom dataset and the published dataset, respectively, are available for download at (https://pan.baidu.com/s/1CZ9vQB22jRQSlaGxGRkpHg?pwd=rxw9)
We are pleased to announce that our paper has been accepted for publication in The International Journal of Applied Earth Observation and Geoinformation. The complete code has been uploaded.
The annotated UV dataset can be downloaded from [...](https://pan.baidu.com/s/1D1LuZQiZSGipq9v7njAppA?pwd=8djc)

> Official implementation of “UVSENet: A Semantic-Embedded dual branch fusion network for urban villages mapping”
> [International Journal of Applied Earth Observation and Geoinformation], [2026]
[![Paper]([![Paper](https://img.shields.io/badge/Paper-PDF-red)]([[Paper Link](https://authors.elsevier.com/sd/article/S1569-8432(26)00353-5)]))]



# Environment Setup
Requirements
Python: [> 3.8]
PyTorch: [> 1.8.1]
CUDA: [ > 11.1]
Operating System: [Linux]

## Environment Setup

### Requirements

* Python: `[e.g., 3.10]`
* PyTorch: `[e.g., 2.3.1]`
* CUDA: `[e.g., 12.1]`
* Operating System: `[e.g., Ubuntu 20.04 / Windows 11]`

### Installation

Create and activate a Conda environment:

```bash
conda create -n [environment_name] python=[python_version]
conda activate [environment_name]
```

Clone this repository:

```bash
git clone [repository_url]
cd [project_name]
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Install PyTorch manually when necessary:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu[CUDA_VERSION]
```

Verify the installation:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

Expected output:

```text
[PyTorch version]
True
```

---

## Dataset Preparation

### Dataset Download

Download the dataset from:

* Dataset: [Dataset name]
* Download link: [Dataset link]
* Access instructions: [Required registration / password / agreement, if applicable]

### Dataset Structure

Organize the dataset as follows:

```text
data/
├── train/
│   ├── images/
│   └── labels/
├── val/
│   ├── images/
│   └── labels/
└── test/
    └── images/
```

### Data Preprocessing

Run the preprocessing script:

```bash
python tools/preprocess.py \
    --input_dir data/raw \
    --output_dir data/processed
```

For custom datasets, ensure that:

* Images are stored in `[image format, e.g., JPG / PNG / TIFF]`.
* Labels are stored in `[label format, e.g., PNG / JSON / TXT]`.
* Image-label filenames are matched correctly.
* Dataset paths are updated in `[config file path]`.

---

## Pretrained Models

Download pretrained weights from:

* [Model name]: [Download link]

Place the checkpoint file in:

```text
checkpoints/
└── [checkpoint_name].pth
```

Example:

```bash
mkdir -p checkpoints
wget [checkpoint_download_link] -O checkpoints/[checkpoint_name].pth
```

---

## Training

Train the model using the default configuration:

```bash
python train.py \
    --config configs/[config_name].yaml \
    --output_dir outputs/[experiment_name]
```

For multi-GPU training:

```bash
torchrun --nproc_per_node=[number_of_gpus] train.py \
    --config configs/[config_name].yaml \
    --output_dir outputs/[experiment_name]
```

Important arguments:

| Argument       | Description                        |
| -------------- | ---------------------------------- |
| `--config`     | Path to the configuration file     |
| `--output_dir` | Directory for logs and checkpoints |
| `--batch_size` | Training batch size                |
| `--epochs`     | Number of training epochs          |
| `--lr`         | Learning rate                      |
| `--resume`     | Resume training from a checkpoint  |

Example:

```bash
python train.py \
    --config configs/[config_name].yaml \
    --batch_size 8 \
    --epochs 100 \
    --lr 0.0001 \
    --output_dir outputs/experiment_01
```

---

## Evaluation

Evaluate a trained model:

```bash
python test.py \
    --config configs/[config_name].yaml \
    --checkpoint checkpoints/[checkpoint_name].pth \
    --data_dir data/test
```

Evaluation results will be saved in:

```text
outputs/
└── [experiment_name]/
    ├── metrics.txt
    ├── predictions/
    └── visualizations/
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

