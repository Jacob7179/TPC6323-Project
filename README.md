<h1 align="center">TPC6323-Project</h1>

## 📋 Project Overview

**Title:** Parallel Image Segmentation using K-Means for Brain Tumour

**Category:** Signal Processing

This project implements and compares five brain tumor segmentation algorithms (K-Means, Mean Shift, Fuzzy C-Means, SLIC Superpixels, and Graph-Based Segmentation) across three different execution models: sequential CPU, CPU-parallel, and GPU-parallel (using ILGPU). The primary objective is to measure computational speedups without modifying the underlying algorithms, providing practical insights for real-time brain tumor diagnosis applications.

## ✨ Key Features

- **Five Segmentation Algorithms:** K-Means, Mean Shift, Fuzzy C-Means, SLIC, and Graph-Based
- **Three Execution Models:** Sequential CPU, Multi-threaded CPU (Parallel.For), GPU Parallel (ILGPU)
- **Comprehensive Performance Benchmarking:** Using BenchmarkDotNet for accurate measurements
- **End-to-End Pipeline:** From image preprocessing to segmentation mask generation
- **Dataset:** Kaggle Brain Tumor MRI Dataset with glioma, pituitary, and meningioma classes

## 🛠️ Technologies Used

- **Language:** C#
- **GPU Acceleration:** ILGPU
- **Benchmarking:** BenchmarkDotNet

## 📊 Dataset Used
- 
- 

## 📁 Project Structure

- `Parallel GBS` - Folder for Graph Based Segmentation (CPU, GPU & Sequential)
- 
- 
- 

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