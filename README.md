# UMC-ICLA: An Unsupervised Multimodal Clustering Model via Indirect Contrastive Learning and Semantic Alignment

## Introduction

Multimodal clustering is important for discovering complex semantics from text, video, and audio. However, existing methods still face two major challenges: modality heterogeneity and the distribution inconsistency between original and augmented data.

## Dependencies

We use Anaconda to create the Python environment and install the required libraries:

```bash
conda create --name umc-icla python=3.8
conda activate umc-icla

pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt
```

## Datasets

- MIntRec: the first multimodal intent recognition dataset ([Paper](https://dl.acm.org/doi/10.1145/3503161.3547906), [Resource](https://github.com/thuiar/MIntRec))
- MELD-DA: a multimodal multi-party dataset for emotion recognition in conversations ([Paper](https://aclanthology.org/P19-1050/), [Resource](https://github.com/declare-lab/MELD))
- IEMOCAP-DA: the Interactive Emotional Dyadic Motion Capture database ([Paper](https://link.springer.com/article/10.1007/s10579-008-9076-6), [Resource](https://sail.usc.edu/iemocap/))

For MELD-DA and IEMOCAP-DA, we use the dialogue-act labels provided by the EMOTyDA dataset ([Paper](https://aclanthology.org/2020.acl-main.402/), [Resource](https://github.com/sahatulika15/EMOTyDA)).

## Features Preparation

Please visit the [official UMC repository](https://github.com/thuiar/UMC#features-preparation) to find the download links for the processed multimodal features and pretrained BERT model.

The expected directory structure is:

```text
Datasets/
├── MIntRec/
│   ├── train.tsv
│   ├── dev.tsv
│   ├── test.tsv
│   ├── video_data/
│   │   └── swin_feats.pkl
│   └── audio_data/
│       └── wavlm_feats.pkl
├── MELD-DA/
│   └── ...
└── IEMOCAP-DA/
    └── ...
```

Download [`uncased_L-12_H-768_A-12`](https://github.com/thuiar/UMC#features-preparation) by following the instructions in the UMC repository, then place the extracted BERT directory in the repository root, as expected by the configuration files.

## Usage

Run all commands from the repository root. The public method identifier is `umc-icla`; Python module names use `UMC_ICLA` because hyphens are not valid in Python identifiers.

For example, run UMC-ICLA on MIntRec with:

```bash
python run.py \
  --dataset MIntRec \
  --data_path Datasets \
  --method umc-icla \
  --multimodal_method umc-icla \
  --text_backbone bert-base-uncased \
  --config_file_name umc_icla_MIntRec.py \
  --video_feats_path swin_feats.pkl \
  --audio_feats_path wavlm_feats.pkl \
  --results_file_name results_umc_icla.csv \
  --output_path outputs/MIntRec \
  --seed 0 \
  --gpu_id 0 \
  --train \
  --tune \
  --save_model \
  --save_results
```

Use the corresponding configuration file for each dataset:

| Dataset | Configuration |
| --- | --- |
| MIntRec | `configs/umc_icla_MIntRec.py` |
| MELD-DA | `configs/umc_icla_MELD-DA.py` |
| IEMOCAP-DA | `configs/umc_icla_IEMOCAP-DA.py` |

To reproduce experiments with multiple random seeds, run the command with seeds `0`, `1`, `2`, `3`, and `4`.

## Acknowledgments

This codebase is built upon the official [UMC](https://github.com/thuiar/UMC) implementation. We thank the UMC authors and the authors of all compared methods for releasing their code and resources.
