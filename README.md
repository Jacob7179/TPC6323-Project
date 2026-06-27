<h1 align="center">TPC6323-Project</h1>

## 📋 Project Overview

**Subject:** TPC6323 Parallel Computing

**Title:** Parallel Image Segmentation using K-Means for Brain Tumour

**Category:** Signal Processing

This project implements and compares five brain tumor segmentation algorithms (K-Means, Mean Shift, Fuzzy C-Means, SLIC Superpixels, and Graph-Based Segmentation) across three different execution models: sequential CPU, CPU-parallel, and GPU-parallel (using ILGPU). The primary objective is to measure computational speedups without modifying the underlying algorithms, providing practical insights for real-time brain tumor diagnosis applications.

## ✨ Key Features

- **Five Segmentation Algorithms:** K-Means, Mean Shift, Fuzzy C-Means, SLIC, and Graph-Based
- **Three Execution Models:** Sequential CPU, Multi-threaded CPU (Parallel.For), GPU Parallel (ILGPU)
- **Comprehensive Performance Benchmarking:** Using BenchmarkDotNet for accurate measurements
- **End-to-End Pipeline:** From image preprocessing to segmentation mask generation
- **Dataset:** Kaggle Brain Tumor MRI Dataset with glioma, pituitary, and meningioma classes

## 🛠️ Technology Used

- **Language:** C#
- **GPU Acceleration:** ILGPU
- **Benchmarking:** BenchmarkDotNet

## 📊 Dataset Used
- https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset
- https://www.kaggle.com/code/noorsaeed/99-mri-classification-with-grad-cam-segmentation

The datasets include MRI scans of:
- Glioma
- Meningioma
- Pituitary

## 📁 Project Structure
```
TPC6323-Project/
│
├── SLIC/                                               # SLIC Superpixels segmentation
│   ├── Benchmark-focusOnAlgOnly/                       # BenchmarkDotNet performance evaluation (CPU, GPU, Sequential & Prototype)
│   ├── Dataset/                                        # Brain MRI datasetssegmentation (glioma, meningioma & pituitary)
│   ├── Grad-CAM-based pseudo ground truth/             # Pseudo ground truth generated using Grad-CAM (glioma, meningioma & pituitary)
│   ├── Parallel Slic (Full Benchmark Check)/           # Parallel SLIC (CPU & GPU)
│   ├── Prototype/K-Means-GPU                           # Prototype
│   └── TPC Preprocessing/                              # Image preprocessing utilities (Dataset, Preprocess Dataset & Preprocessing Code)
│
├── Fuzzy_C_Means/                                      # Fuzzy C-Means  
│   ├── TPC_Fuzzy_CPU                                   # CPU Parallel
│   └── TPC_Fuzzy_GPU                                   # GPU Parallel 
│
├── GBS                                                 # Graph-Based Segmentation
│   ├── backup/                                         # Backup source codes
│   ├── GBS_CPU/                                        # CPU Parallel
│   ├── GBS_GPU/                                        # GPU Parallel implementation (ILGPU)
│   └── Sequential GBS/                                 # Sequential
│
└──
```

## 👥 Contributors

<table>
    <tbody>
        <tr>
            <td align="center" valign="top" width="33.33%">
                <a href="https://github.com/Jacob7179" target="_blank">
                    <img src="https://avatars.githubusercontent.com/u/70430960?v=4" width="100px;" alt="Jacob7179 Avatar"/><br />
                    <sub><b>Jacob7179</b></sub>
                </a>
            </td>
            <td align="center" valign="top" width="33.33%">
                <a href="https://github.com/jeremypangdv" target="_blank">
                    <img src="https://avatars.githubusercontent.com/u/168976310?v=4" width="100px;" alt="jeremypangdv Avatar"/><br />
                    <sub><b>jeremypangdv</b></sub>
                </a>
            </td>
            <td align="center" valign="top" width="33.33%">
                <a href="https://github.com/t1an-wei" target="_blank">
                    <img src="https://avatars.githubusercontent.com/u/242118479?v=4" width="100px;" alt="t1an-wei Avatar"/><br />
                    <sub><b>t1an-wei</b></sub>
                </a>
            </td>
        </tr>
    </tbody>
</table>