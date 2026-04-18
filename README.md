<h1 align="center">
  Human Body Restoration with <br>
  One-Step Diffusion Model and A New Benchmark
</h1>

<p align="center">
🚩 <strong>Accepted by ICML 2025</strong>
</p>

<p align="center">
<a href="https://github.com/gobunu">Jue Gong</a>,
<a href="https://github.com/jkwang28">Jingkai Wang</a>,
<a href="https://zhengchen1999.github.io/">Zheng Chen</a>,
Xing Liu, Hong Gu,
<a href="http://yulunzhang.com/">Yulun Zhang</a>,
<a href="https://scholar.google.com/citations?user=yDEavdMAAAAJ">Xiaokang Yang</a>
</p>

<p align="center">
"A new benchmark and the first one-step diffusion model for human body restoration.", 2025
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2502.01411">
    <img src="https://img.shields.io/badge/Paper-arXiv-red?logo=arxiv&logoSvg">
  </a>
  <a href="https://github.com/gobunu/OSDHuman/releases/download/v1/supp.pdf">
    <img src="https://img.shields.io/badge/Supplementary_material-Paper-orange.svg">
  </a>
  <a href="https://github.com/gobunu/OSDHuman/releases">
    <img src="https://img.shields.io/github/downloads/gobunu/OSDHuman/total.svg">
  </a>
  <a href="https://github.com/gobunu/OSDHuman">
    <img src="https://visitor-badge.laobi.icu/badge?page_id=gobunu.OSDHuman&right_color=violet">
  </a>
  <a href="https://github.com/gobunu/OSDHuman">
    <img src="https://img.shields.io/github/stars/gobunu/OSDHuman?style=social">
  </a>
</p>

#### 🔥🔥🔥 News

- **2025-02-05:** The OSDHuman repository was released.
- **2025-05-01:** OSDHuman was accepted to **ICML 2025**.
- **2025-05-04:** Released the PERSONA test and validation sets.
- **2026-04-18:** Released the public inference code and inference instructions.

---

> **Abstract:** Human body restoration, as a specific application of image restoration, is widely applied in practice and plays a vital role across diverse fields. However, thorough research remains difficult, particularly due to the lack of benchmark datasets. In this study, we propose a high-quality dataset automated cropping and filtering (HQ-ACF) pipeline. This pipeline leverages existing object detection datasets and other unlabeled images to automatically crop and filter high-quality human images. Using this pipeline, we constructed a person-based restoration with sophisticated objects and natural activities (PERSONA) dataset, which includes training, validation, and test sets. The dataset significantly surpasses other human-related datasets in both quality and content richness. Finally, we propose OSDHuman, a novel one-step diffusion model for human body restoration. Specifically, we propose a high-fidelity image embedder (HFIE) as the prompt generator to better guide the model with low-quality human image information, effectively avoiding misleading prompts. Experimental results show that OSDHuman outperforms existing methods in both visual quality and quantitative metrics.

![](images/OSDHuman_overall.png)

---

[<img src="images/test-1.png" height="200"/>](https://imgsli.com/MzQ1ODk0)
[<img src="images/test-2.png" height="200"/>](https://imgsli.com/MzQ1ODky)
[<img src="images/Val-1.png" height="200"/>](https://imgsli.com/MzQ1ODk3)
[<img src="images/Val-2.png" height="200"/>](https://imgsli.com/MzQ1ODk4)

---

## ⚒️ TODO

* [x] Release inference code
* [x] Release PERSONA test and validation sets
* [ ] Release training code
* [ ] Release PERSONA training set

## 🔗 Contents

- [x] [Inference](#inference)
- [x] [PERSONA Test and Validation Set](#test_validation_set)
- [x] [Results](#results)
- [x] [Citation](#citation)
- [ ] [Acknowledgements](#acknowledgements)

## <a name="inference"></a>🚀 Inference

This repository provides the public OSDHuman inference code. The entry point is [`run_inference.py`](run_inference.py), and sample inputs are included under [`test_set/`](test_set).

### Environment Installation

```bash
conda create -n osdhuman python=3.10 -y
conda activate osdhuman
pip install --upgrade pip
pip install -r requirements.txt
```

### Prepare Checkpoints

Create local runtime folders if needed:

```bash
mkdir -p weights preset outputs
```

Place the following OSDHuman checkpoints under `weights/`:

- `weights/model.pkl`
- `weights/model_embedding_change.pkl`

Download source:

- Google Drive folder: <https://drive.google.com/drive/folders/1sNoSNcJC7cdf0RR5dK87PVTEupjOZ-3B?usp=drive_link>

Place the following external base models locally:

- `preset/stable-diffusion-2-1-base`
  Source: <https://huggingface.co/stabilityai/stable-diffusion-2-1-base>
- `preset/ram_swin_large_14m.pth`
  Source: <https://huggingface.co/spaces/xinyu1205/recognize-anything/blob/main/ram_swin_large_14m.pth>

### Quick Start

Run inference on the bundled sample set:

```bash
python run_inference.py \
  -i test_set \
  -o outputs \
  --osediff_path weights/model.pkl \
  --pretrained_model_name_or_path preset/stable-diffusion-2-1-base \
  --ram_path preset/ram_swin_large_14m.pth \
  --ram_img_only
```

`run_inference.py` accepts either a single image path or a directory and writes restored images into `outputs/` by default.

## <a name="test_validation_set"></a>🎭 PERSONA Test and Validation Set

We provide three sets for evaluation, including both high-quality and degraded validation images and a real-world test set.

| Dataset | Description | Download Link |
|---|---|---|
| PERSONA-Val HQ | High-quality validation set | [Google Drive](https://drive.google.com/file/d/1JaUzyDBJHytl3MfFBdcGAaO7uYDfP3vS/view?usp=drive_link) |
| PERSONA-Val LQ | Low-quality validation set | [Google Drive](https://drive.google.com/file/d/1qzJETavxMFxUDyimYxg4Ob5GmAAHmuMG/view?usp=drive_link) |
| PERSONA-Test | Real-world test set | [Google Drive](https://drive.google.com/file/d/1n_rywHB2apB9kWgZSRdnzMxQsJKWIEMR/view?usp=drive_link) |

## <a name="results"></a>🔎 Results

The model **OSDHuman** achieved state-of-the-art performance on both **PERSONA-Val** and **PERSONA-Test**. Detailed comparisons are available in the paper.

<details>
<summary>&ensp;Quantitative Comparisons (click to expand)</summary>
<li>Results in Table 2 on synthetic PERSONA-Val and real-world PERSONA-Test datasets from the main paper.</li>
<p align="center">
<img src="images/table2-main.png">
</p>
</details>

<details open>
<summary>&ensp;Visual Comparisons (click to expand)</summary>
<li>Results in Figure 8 on the real-world PERSONA-Test dataset from the main paper.</li>
<p align="center">
<img src="images/figure8-main.png">
</p>
<li>Results in Figure 9 on the synthetic PERSONA-Val dataset from the main paper.</li>
<p align="center">
<img src="images/figure9-main.png">
</p>
</details>

<details>
<summary style="margin-left: 2rem;">&ensp;More Comparisons on real-world PERSONA-Test dataset...</summary>
<li style="margin-left: 2rem;">Results in Figures 3 and 4 from the supplementary material.</li>
<p align="center">
<img src="images/figure3-supp.png">
</p>
<p align="center">
<img src="images/figure4-supp.png">
</p>
</details>

<details>
<summary style="margin-left: 2rem;">&ensp;More Comparisons on synthetic PERSONA-Val dataset...</summary>
<li style="margin-left: 2rem;">Results in Figures 5 and 6 from the supplementary material.</li>
<p align="center">
<img src="images/figure5-supp.png">
</p>
<p align="center">
<img src="images/figure6-supp.png">
</p>
</details>

## <a name="citation"></a>📎 Citation

If you find the code helpful in your research or work, please cite:

```bibtex
@inproceedings{gong2025osdhuman,
    title={Human Body Restoration with One-Step Diffusion Model and A New Benchmark},
    author={Gong, Jue and Wang, Jingkai and Chen, Zheng and Liu, Xing and Gu, Hong and Zhang, Yulun and Yang, Xiaokang},
    booktitle={ICML},
    year={2025}
}
```

## <a name="acknowledgements"></a>💡 Acknowledgements

[OSEDiff](https://github.com/cswry/OSEDiff)
