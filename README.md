<p align="center">
  <a href="" rel="noopener">
 <img width=200px height=200px src=https://raw.githubusercontent.com/gaetandi/cheXpert/85e1ad5f466b41b947f67134105eafebe313bc32//view1_frontal.jpg  alt="Project logo"></a>
  <img width=200px height=200px src=https://stanfordmlgroup.github.io/competitions/chexpert/img/logo.jpg alt="Project logo"></a>
</p>

<h3 align="center">Unsupervised Structured Report Generation via CycleGAN </h3>

<div align="center">

[![Status](https://img.shields.io/badge/status-active-success.svg)]()
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](/LICENSE)

</div>

---

## 📝 Table of Contents

- [About](#about)
- [Getting Started](#getting_started)
- [Usage](#usage)
- [Methodology](#methodology)
- [References](#authors)
- [Authors](#authors)

## 🧐 About

Structured reporting is an important toolin radiology, as it can provideconcise, standardized and complete reports.However, the development of automated systems to generate these reports is hindered by the scarcity of datasets comprising paired radiological images and corresponding structured reports. This project proposes anovel approach to circumvent the need for such paired data by harnessing unsupervised learning techniques, specifically Cycle-Consistent Adversarial Networks (CycleGANs)or similar frameworks, for the generation of coherent and contextually accurate radiological reports from unpaired data.

## 🏁 Getting Started 

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

What things you need to install the software and how to install them.

```bash
git clone <repository-url>
cd Unsupervised-Structured-Reporting-via-Cycle-Consistency
pip install -r requirements.txt
```

### Dataset
[MIMIC-CXR](https://physionet.org/content/mimic-cxr/2.0.0/) dataset used in this project. Follow the link to download it to your local.

* Download the data.

### Update config.yaml and environment_settings.py
Update the config files based on your preferences and give the related paths in the environment settings.

#### Dataset
```bash
dataset:
  chexpert_labels: [...]
  data_imputation: "zeros"
  paired: False
  use_all_images: False
  num_images: 50000
```
* chexpert_labels: List of labels for the CheXpert dataset.
* data_imputation: Strategy for handling missing data.
* paired: Whether the dataset contains paired images.
* use_all_images: Flag to use all images in the dataset.
* num_images: Total number of images to use from the dataset.

#### Image Generator Configuration
Defines settings for the image generator component.
```bash
report_generator:
  model: "biovil_t"
  ...
  consistency_loss: 'BCE'
```
Parameters include model type, learning rate, optimizer, and various model-specific hyperparameters.


#### Report Generator Configuration
Defines settings for the report generator component.
```bash
image_generator:
  model: "cgan"
  ...
  consistency_loss: "biovil_t"
```
Parameters include model type, learning rate, optimizer, and various model-specific hyperparameters.


## Usage

```bash
python3 train.py
jupyter-notebook eval.ipynb
```

## Methodology
![Result Image 2](https://github.com/ardamamur/Unsupervised-Anomaly-Detection-in-Medical-Images-via-PaDiM/blob/main/sources/network.png?raw=true)


## References


## Authors
* [Arda Mamur](https://github.com/ardamamur) [Maximilian Oberle](https://github.com/ardamamur) [Zhiang Guo](https://github.com/ardamamur) [Furqan Lodhi](https://github.com/ardamamur) 
