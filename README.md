# RIMD-DLAC-SPECT-MPI
Reconstruction-Informed and Multidomain Deep Learning for Generalizable CT-Free Attenuation Correction in SPECT Myocardial Perfusion Imaging

End-to-end pipeline for SPECT attenuation correction (AC) with training, multi-input inference, and ensemble ATM → MAC reconstruction.

## Repository Structure
- `models/` – trained model files used for inference
- `train.py` – training entry point
- `inference_external.py` – legacy inference/evaluation utilities
- `inference_test.py` – production inference pipeline (multi-input + ensemble + NAC/ATM/MAC saving)
- `evaluate.py` – metrics and model evaluation helpers
- `image_evaluation.py` - calculate image evaluation metric
- `utils.py` – shared transforms and IO helpers
- `requirements.txt` – pinned runtime dependencies

## Environment


```bash
# create and activate conda env
conda create -n rimd python=3.10 -y
conda activate rimd

# install runtime deps
pip install -r requirements.txt
```

If you need GPU-enabled PyTorch, install it before the requirements (follow the official PyTorch selector for the right cudatoolkit), e.g.:

```bash
conda install pytorch torchvision torchaudio cudatoolkit=11.8 -c pytorch -c conda-forge
pip install -r requirements.txt
```
- Python packages are pinned in `requirements.txt`.
- Install with:
```bash
pip install -r requirements.txt
```

## Training
`train.py` is the training entry point. It reads paired input reconstructions and target attenuation maps, trains the model, saves checkpoints and logs to the experiment directory, and exports the best model (.pt).

Required / important flags (used in the example):
- --data_path_train: root folder containing training data (see "Data layout" below).
- --exp_dir: base output directory where experiment folders, checkpoints and logs are written.
- --exp: experiment name (subfolder under --exp_dir).
- --input_type: input modality descriptor (e.g., OSEM_3 or a single OSEM variant).
- --num_input: how many input channels (1 or 3 for multi-input with OSEM_4I4S/OSEM_6I6S/OSEM_8I8S).

Common optional flags (may vary by implementation):
- --batch_size, --epochs, --lr, --num_workers
- --resume (path to checkpoint to continue training)
- --pretrained (use a pretrained backbone)
Check the header of train.py for the exact supported flags.

Example:
```bash
python train.py \
  --data_path_train /path/to/train \
  --exp_dir /path/to/output \
  --exp my_train_run \
  --input_type OSEM_3 \
  --num_input 3
```

What to expect in output
- Checkpoints and training logs under: <exp_dir>/<exp>/<input_type>
- Best model saved as a .pt file (name depends on train.py implementation).
- TensorBoard (if enabled) logs in the experiment folder.

Data layout and expectations for training
- All NIfTI files should be .nii.gz. NM inputs are DICOM (.dcm) or DICOM series directories.
- Filenames across modalities must correspond (e.g., 00000.nii.gz in ATM matches 00000.nii.gz in each MAC/NAC subfolder for that case).

Recommended folder structure under --data_path_train:
```
/path/to/train/
├── ATM/                       # target attenuation maps (NIfTI)
│   ├── 00000.nii.gz
│   ├── 00001.nii.gz
│   └── ...
├── MAC/                       # attenuation-corrected reconstructions (NIfTI)
│   ├── OSEM_4I4S/
│   │   ├── 00000.nii.gz
│   │   └── ...
│   ├── OSEM_6I6S/
│   │   ├── 00000.nii.gz
│   │   └── ...
│   └── OSEM_8I8S/
│       ├── 00000.nii.gz
│       └── ...
├── NAC/                       # non-attenuation-corrected reconstructions (NIfTI)
│   ├── OSEM_4I4S/
│   │   ├── 00000.nii.gz
│   │   └── ...
│   ├── OSEM_6I6S/
│   │   ├── 00000.nii.gz
│   │   └── ...
│   └── OSEM_8I8S/
│       ├── 00000.nii.gz
│       └── ...
└── NM/                        # raw NM DICOM inputs (per study)
    ├── 00000.dcm  (or directory with DICOM files for study 00000)
    ├── 00001.dcm
    └── ...
```

Multi-input training notes
- For num_input=3, the model expects NAC inputs from OSEM_4I4S, OSEM_6I6S and OSEM_8I8S as three aligned channels. Ensure each OSEM_* subfolder contains the same set of case IDs.
- ATM targets must align with NAC/MAC case IDs.

If the repo's train.py supports other options (augmentation, patching, normalization, custom samplers), consult the script's argument parser or top-of-file documentation to match flags to your desired behavior.
Example (adjust paths/flags per your setup):
```bash
python train.py \
  --data_path_train /path/to/train/NIfTI \
  --exp_dir /path/to/output \
  --exp my_train_run \
  --input_type OSEM_3 \
  --num_input 3
```

### Inference with `inference_external.py` 

`inference_external.py` is designed for running inference on external test datasets using trained models when labels are available. It supports single-input and multi-input (3-channel) configurations and can ensemble predictions from multiple models. This script expects your data to be organized in the same folder structure as used for training (see above), with subfolders for ATM, MAC, NAC, and NM. 
The output directory used for inference must be the same as the one used during training.
#### Example Usage

To run inference on an external test dataset:

```bash
python inference_external.py \
  --nm_raw_dir /path/to/external/NM \
  --output_dir /path/to/output \
  --exp_name my_external_test \
  --model_dir /path/to/models_dir \
  --colimator G8-LEHR \
  --num_workers 4
```

- `--nm_raw_dir`: Directory containing raw NM DICOM files for external studies.
- `--output_dir`: Output directory for results.
- `--exp_name`: Name for this inference run.
- `--model_dir`: Directory containing trained `.pt` model files.
- `--colimator`: Specify colimator type (e.g., SY-LEHR).
- `--num_workers`: Number of parallel workers for processing.

The script will process the external NM data, generate NAC, ATM, and MAC reconstructions, and save outputs in the specified structure.


## Inference (Multi-Input + Ensemble) with `inference_test.py`
- This file is for using the model when labels are not available.
- Multi-input only (3 channels: OSEM_4I4S, OSEM_6I6S, OSEM_8I8S).
- ATM predictions are generated by all models in `--model_dir`, ensembled, and individual model folders are removed after ensembling.
- All ATM/MAC/NAC/NM saving uses in the output folder.

Run:
```bash
python inference_test.py \
  --nm_raw_dir /path/to/NM \
  --output_dir /path/to/output \
  --exp_name my_experiment \
  --model_dir /path/to/models_dir \
  --colimator SY-LEHR \
  --num_workers 4
```

Notes:
- Place your trained `.pt` files in `--model_dir` (ensemble expects multiple).
- The pipeline:
  1) NM → NAC (OSEM 4/6/8) with DICOM save → NIfTI convert → DICOM cleanup
  2) NAC → ATM predictions (per model) → ATM ensemble (average) → per-model folders removed
  3) ATM ensemble + NM raw → MAC reconstructions (OSEM 4/6/8)

#### Input Structure

- Input NM raw directory contains scanner raw files (per study/exam). Example:
```
/path/to/NM/
  ├── 00000.dcm
  ├── 00001.dcm
  └── ...
```

## Output Structure
`inference_test.py` creates:
```
output_dir/exp_name/
├── NM/                    # Copied NM raw files
├── NAC/                   # NAC reconstructions (NIfTI)
│   ├── OSEM_4I4S/
│   ├── OSEM_6I6S/
│   └── OSEM_8I8S/
├── ATM/          # Ensemble ATM predictions (averaged)
└── MAC/                   # Final AC reconstructions (NIfTI)
    ├── OSEM_4I4S/
    ├── OSEM_6I6S/
    └── OSEM_8I8S/
```

## License

This project is licensed under the Apache License, Version 2.0 (SPDX: Apache-2.0). See the included LICENSE file for full terms.

## Acknowledgements

This project uses and acknowledges the following third‑party software:

- MONAI — Medical Open Network for AI (https://github.com/Project-MONAI). Licensed under Apache‑2.0.
- pytomography / SPECT reconstruction utilities (https://github.com/pytomography). Licensed under MIT.

## Citation

Hajianfar G, et al. Reconstruction-Informed and Multidomain Deep Learning for Generalizable CT-Free Attenuation Correction in SPECT Myocardial Perfusion Imaging. In submission.
Ghasem Hajianfar, et al. Reconstruction-informed and multidomain deep learning for generalizable CT-free attenuation correction in SPECT myocardial perfusion imaging, Medical Image Analysis, Volume 113, 2026, 104220, ISSN 1361-8415, https://doi.org/10.1016/j.media.2026.104220.

```bibtex
@article{HAJIANFAR2026104220,
title = {Reconstruction-informed and multidomain deep learning for generalizable CT-free attenuation correction in SPECT myocardial perfusion imaging},
journal = {Medical Image Analysis},
volume = {113},
pages = {104220},
year = {2026},
issn = {1361-8415},
doi = {https://doi.org/10.1016/j.media.2026.104220},
url = {https://www.sciencedirect.com/science/article/pii/S1361841526002896},
author = {Ghasem Hajianfar and Yazdan Salimi and Mehdi Amini and Xiaotong Hong and René Nkoulou and Elnaz Jenabi and Zahra Mansouri and Atena Aghaee and Soroush Bagheri and Amirhossein Sanaat and Ahmad Bitarafan-Rajabi and Hossein Arabi and Isaac Shiri and Habib Zaidi},
keywords = {SPECT, Myocardial perfusion imaging, Attenuation correction, Deep learning, Multidomain},
}
```
