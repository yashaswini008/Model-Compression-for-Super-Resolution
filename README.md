# Model Compression for Super Resolution

##  Overview
This project focuses on enhancing low-resolution images using deep learning-based **Super Resolution models (SRCNN & FSRCNN)** and optimizing them for real-time performance using **model compression techniques**.

Super-resolution is widely used in applications like:
- Video conferencing
- Image enhancement
- Computer vision tasks

 Goal: Improve image quality while reducing computational cost for real-time CPU usage.

---

##  Problem Statement
Deep learning super-resolution models provide high-quality outputs but suffer from:
- High computational complexity
- Large memory usage
- Poor real-time performance on CPUs

 This project solves this by building **lightweight and efficient models** using compression techniques. :contentReference[oaicite:0]{index=0}

---

##  Approach & Architecture


::contentReference[oaicite:1]{index=1}


The pipeline includes:
1. Dataset collection
2. Model selection (SRCNN, FSRCNN)
3. Model training & retraining
4. Model compression (Quantization)
5. Evaluation (PSNR, SSIM)
6. Deployment optimization using ONNX

---

##  Datasets Used

### 1. FFHQ Dataset
- 70,000 high-quality face images
- Resolution: 1024×1024
- Used for learning fine facial details :contentReference[oaicite:2]{index=2}

### 2. WebScreenshots Dataset
- 20,000 website screenshots
- Multilingual dataset
- Used for screen content enhancement :contentReference[oaicite:3]{index=3}

---

## ⚙️ Data Preprocessing
- Generated **Low Resolution (LR)** images using bicubic downsampling
- Created LR–HR image pairs
- Converted images to **YCbCr format** (used luminance channel)

---

##  Models Used

###  SRCNN (Super Resolution CNN)
- Upsamples image first using bicubic interpolation
- Learns mapping from LR → HR
- 3-layer CNN architecture

###  FSRCNN (Fast SRCNN)
- Takes LR image directly
- Uses deconvolution for upsampling
- Faster and more efficient than SRCNN :contentReference[oaicite:4]{index=4}

---

##  Evaluation Metrics

###  PSNR (Peak Signal-to-Noise Ratio)
- Measures image quality
- Higher value = better quality

###  SSIM (Structural Similarity Index)
- Measures similarity between images
- Value closer to 1 = better similarity :contentReference[oaicite:5]{index=5}

---

##  Results & Performance

###  Training vs Validation Loss

::contentReference[oaicite:6]{index=6}


---

###  PSNR Performance

::contentReference[oaicite:7]{index=7}


---

###  SSIM Performance

::contentReference[oaicite:8]{index=8}


---

##  Output Results (Before vs After)


::contentReference[oaicite:9]{index=9}


👉 The model successfully:
- Enhanced image clarity
- Reduced blur
- Preserved fine details

---

##  Model Compression Techniques

###  Quantization
- Reduces model precision (FP32 → INT8)
- Improves speed and reduces memory usage :contentReference[oaicite:10]{index=10}

###  ONNX Runtime
- Optimized inference engine
- Enables fast CPU execution
- Cross-framework compatibility :contentReference[oaicite:11]{index=11}

---

##  Tech Stack
- Python
- PyTorch
- NumPy, OpenCV
- ONNX Runtime
- Matplotlib

---

##  Applications
- Video conferencing enhancement
- Low bandwidth video streaming
- Image restoration
- Real-time computer vision systems

---

##  Conclusion
This project successfully demonstrates that:
- Deep learning models can enhance image quality effectively
- Model compression enables real-time performance
- FSRCNN provides faster inference compared to SRCNN

 Achieved a balance between **accuracy and efficiency** suitable for real-world applications. :contentReference[oaicite:12]{index=12}

