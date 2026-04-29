# Model Compression for Image Super-Resolution

## Overview
This project focuses on **Model Compression Techniques for Image Super-Resolution** using Deep Learning.  
The main goal is to reduce the model size and computational complexity while maintaining high image reconstruction quality.

The project applies:
- Pruning
- Quantization
- Knowledge Distillation
- Lightweight CNN Architectures

to optimize Super-Resolution models for efficient deployment on low-resource devices.

---

## Features
- Image Super-Resolution using Deep Learning
- Model Compression Techniques
- Reduced Model Size
- Faster Inference
- Performance Evaluation
- High-Quality Image Reconstruction

---

## Technologies Used
- Python
- TensorFlow / Keras
- NumPy
- OpenCV
- Matplotlib
- Jupyter Notebook

---

## Project Structure

```bash
Model-Compression-for-Super-Resolution/
│
├── datasets/
├── models/
├── outputs/
├── images/
├── notebooks/
├── requirements.txt
└── README.md
```

---

## Model Compression Techniques Used

### 1. Pruning
Removes unnecessary weights from the neural network to reduce model complexity.

### 2. Quantization
Converts floating-point weights into lower precision values for faster inference.

### 3. Knowledge Distillation
Transfers knowledge from a large teacher model to a smaller student model.

### 4. Lightweight Architecture
Uses efficient CNN structures for reduced computation.

---

## Workflow

1. Load Dataset
2. Preprocess Images
3. Train Super-Resolution Model
4. Apply Compression Techniques
5. Evaluate Performance
6. Generate High-Resolution Outputs

---

# Results

## Low Resolution Input
![Low Resolution](images/low_resolution.png)

---

## Super-Resolved Output
![Super Resolution Output](images/output.png)

---

## Ground Truth Image
![Ground Truth](images/ground_truth.png)

---

## Performance Comparison

| Model | PSNR | SSIM | Model Size |
|------|------|------|------|
| Original Model | 32.5 | 0.91 | 120 MB |
| Compressed Model | 31.8 | 0.89 | 35 MB |

---

## Output Visualization

<p align="center">
  <img src="images/comparison.png" width="700"/>
</p>

---

## Installation

Clone the repository:

```bash
git clone https://github.com/yashaswini008/Model-Compression-for-Super-Resolution.git
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the notebook or training script.

---

## Future Improvements
- Deploy using TensorFlow Lite
- Real-time Super-Resolution
- Mobile Optimization
- GAN-based Super-Resolution

---

## Author

**Yashaswini**
- GitHub: [yashaswini008](https://github.com/yashaswini008)

---

## License
This project is open-source and available under the MIT License.
