<h1 align="center">TPC6323-Project</h1>

## 📋 Project Overview

**Subject:** TPC6323 Parallel Computing

**Title:** Parallel Image Segmentation using K-Means for Brain Tumour

**Category:** Signal Processing

This project investigates the use of parallel computing techniques for brain tumour MRI image segmentation. Five segmentation algorithms are implemented and evaluated under three different execution models:

- Sequential CPU
- CPU Parallelism (Parallel.For)
- GPU Parallelism (ILGPU)

The objective is to analyse computational performance improvements achieved through parallel computing while preserving the original segmentation algorithms. Execution time, speedup, scalability, and efficiency are compared to determine the most suitable execution model for medical image processing applications.

---

# 🎯 Objectives

- Implement five brain tumour segmentation algorithms.
- Develop sequential, CPU-parallel, and GPU-parallel versions of each algorithm.
- Measure execution time using BenchmarkDotNet.
- Compare computational speedup between execution models.
- Demonstrate the benefits of parallel computing in medical image segmentation.

---

# 🧠 Algorithms Implemented

| Algorithm | Description |
|-----------|-------------|
| **K-Means** | Clustering algorithm based on pixel intensity similarity. |
| **Mean Shift** | Density-based clustering for image segmentation. |
| **Fuzzy C-Means** | Soft clustering where pixels belong to multiple clusters with different membership values. |
| **SLIC Superpixels** | Generates superpixels to simplify image segmentation. |
| **Graph-Based Segmentation** | Segments images by modelling pixels as graph nodes and minimizing edge weights. |

---

# ⚡ Execution Models

| Execution Model | Description |
|-----------------|-------------|
| **Sequential CPU** | Traditional single-threaded implementation. |
| **CPU Parallel** | Multi-threaded implementation using `Parallel.For`. |
| **GPU Parallel** | GPU implementation using **ILGPU** for massive parallel execution. |

---

## ✨ Key Features

- **Five Segmentation Algorithms:** K-Means, Mean Shift, Fuzzy C-Means, SLIC, and Graph-Based
- **Three Execution Models:** Sequential CPU, Multi-threaded CPU (Parallel.For), GPU Parallel (ILGPU)
- **Comprehensive Performance Benchmarking:** Using BenchmarkDotNet for accurate measurements
- **End-to-End Pipeline:** From image preprocessing to segmentation mask generation
- **Dataset:** Kaggle Brain Tumor MRI Dataset with glioma, pituitary, and meningioma classes

---

# 🛠 Technology Used

| Category | Technology |
|----------|------------|
| Programming Language | C# |
| Framework | .NET |
| GPU Computing | ILGPU |
| Parallel Programming | Task Parallel Library (TPL) |
| Benchmarking | BenchmarkDotNet |
| Image Processing | System.Drawing |

---

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
├── SLIC/                                       # SLIC Superpixels segmentation (Algorithm)
│   ├── Benchmark-focusOnAlgOnly/               # BenchmarkDotNet performance evaluation (CPU, GPU, Sequential & Prototype)
│   ├── Dataset/                                # Brain MRI datasetssegmentation (glioma, meningioma & pituitary)
│   ├── Grad-CAM-based pseudo ground truth/     # Pseudo ground truth generated using Grad-CAM (glioma, meningioma & pituitary)
│   ├── Parallel Slic (Full Benchmark Check)/   # Parallel SLIC (CPU & GPU)
│   └── TPC Preprocessing/                      # Image preprocessing utilities (Dataset, Preprocess Dataset & Preprocessing Code)
│
├── Fuzzy_C_Means/                              # Fuzzy C-Means (Algorithm)
│   ├── TPC_Fuzzy_CPU                           # CPU Parallel
│   └── TPC_Fuzzy_GPU                           # GPU Parallel 
│
├── GBS/                                        # Graph-Based Segmentation (Algorithm)
│   ├── backup/                                 # Backup source codes
│   ├── GBS_CPU/                                # CPU Parallel
│   ├── GBS_GPU/                                # GPU Parallel
│   └── Sequential GBS/                         # Sequential (old)
│
├── K-Means/                                    # K-Means segmentation (Algorithm)
│   ├── Baseline K-Means                        # Sequential CPU
│   ├── CPU-Parallelism K-Means                 # CPU Parallel
│   ├── GPU-Parallelism K-Means                 # GPU Parallel
│   └── K-Means Benchmark                       # BenchmarkDotNet performance evaluation
│
├── Mean_Shift/                                 # Mean Shift segmentation (Algorithm)
│   ├──
│   └── 
│
│
└── Prototype/
    └── K-Means-GPU                             # Prototype (K-Means GPU Parallel)
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