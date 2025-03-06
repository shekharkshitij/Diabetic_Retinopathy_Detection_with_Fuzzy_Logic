# 🏥 Diabetic Retinopathy Detection using EfficientNetV2 and Fuzzy Memberships

## **Introduction**
Diabetic Retinopathy (DR) is one of the leading causes of vision impairment and blindness globally. With the rising number of diabetes cases worldwide, early detection and classification of DR can **help prevent severe visual impairment**. 

This project presents a **deep learning-based approach** that leverages **state-of-the-art convolutional neural networks (CNNs)**, specifically **EfficientNetV2B3**, in combination with **Explainable AI techniques (Grad-CAM & Attention Maps)** and **Fuzzy Membership functions** to classify **Diabetic Retinopathy severity** from **fundus images**.

---

## **Problem Statement**
Diabetic Retinopathy (DR) is a progressive disease characterized by damage to the **retinal blood vessels** due to prolonged diabetes. It has different severity levels:

1. **No DR (Normal Eye)**
2. **Mild DR**: Early microaneurysms
3. **Moderate DR**: More extensive microaneurysms and hemorrhages
4. **Severe DR**: Large hemorrhages and blood vessel distortions
5. **Proliferative DR**: Growth of abnormal blood vessels and vision loss

The main challenges in **automated detection** of DR include:
- **Class Imbalance**: Early-stage DR cases are more common than advanced DR cases.
- **High Variability**: Different patients exhibit different symptoms.
- **Explainability**: Most deep learning models are **black-box**, making them difficult to interpret.

Our approach **addresses these challenges** using:
1. **Data Augmentation & MixUp** to handle **class imbalance**.
2. **EfficientNetV2 with Squeeze-and-Excitation (SE) Blocks** for **robust feature extraction**.
3. **Fuzzy Membership Functions** to **model uncertainty** in classification.
4. **Grad-CAM & Attention Mechanisms** for **explainable AI**.

---

## 📂 **Dataset Description**
The dataset used in this study is the **APTOS 2019 Blindness Detection Dataset**, a publicly available dataset on **Kaggle**.

### **Dataset Details**
- **Number of Images**: ~3,662
- **Image Type**: Retinal Fundus Photographs
- **Labels**: 5-Class Classification (0 to 4)
- **File Format**: `.png`

Each image in the dataset is labeled with one of five classes:
| Class | Severity Level | Description |
|-------|--------------|-------------|
| 0 | No DR | Healthy retina, no visible signs of DR. |
| 1 | Mild DR | Small microaneurysms, no vision-threatening changes. |
| 2 | Moderate DR | Hemorrhages, cotton-wool spots. |
| 3 | Severe DR | More extensive hemorrhages, retinal blood vessel distortion. |
| 4 | Proliferative DR | Abnormal new blood vessels, high risk of vision loss. |

### **Dataset Link**
You can access the dataset here: [APTOS 2019 Blindness Detection Dataset](https://www.kaggle.com/c/aptos2019-blindness-detection/data)

---

## 🛠 **Preprocessing Techniques**
Fundus images contain **noise and artifacts** that need to be removed before training a model. The following preprocessing steps were applied:

### **1. Image Resizing**
- All images were resized to **224×224 pixels** to ensure consistency.

### **2. Contrast Limited Adaptive Histogram Equalization (CLAHE)**
- CLAHE enhances local contrast, making blood vessel abnormalities more visible.

### **3. Gaussian Blur**
- Removes high-frequency noise and smooths out artifacts.

### **4. Pixel Normalization**
- Converts image pixel values into the range `[-1, 1]` to speed up convergence.

---

## 🔀 **Data Augmentation Techniques**
To improve the **generalization capability** of the model, multiple augmentation techniques were applied:

1. **Random Flip**: Horizontally and vertically flips images.
2. **Random Rotation**: Rotates images within a **20% range**.
3. **Random Zoom**: Zooms in/out by **20%**.
4. **MixUp Augmentation**: Mixes two images with an interpolation factor.
5. **CutMix Augmentation**: Replaces part of an image with another image.

These techniques help **reduce overfitting** and improve **robustness**.

---

## 🏗 **Deep Learning Model Architecture**
The backbone of our model is **EfficientNetV2B3**, a **highly optimized CNN architecture** designed for efficient feature extraction.

### **1. EfficientNetV2 Backbone**
- Pretrained on **ImageNet** for transfer learning.
- Uses **Depthwise Separable Convolutions** to improve efficiency.

### **2. Squeeze-and-Excitation (SE) Block**
- Improves feature representation by **dynamically recalibrating feature maps**.

### **3. Channel & Spatial Attention**
- **Channel Attention**: Identifies which feature maps are most important.
- **Spatial Attention**: Highlights important regions in an image.

### **4. Fuzzy Membership Layer**
- Instead of a **hard classification**, this layer **assigns probabilities to multiple classes**, capturing **uncertainty** in classification.

---

## 📊 **Explainability: Grad-CAM & Attention Maps**
One of the biggest challenges in **medical AI** is the **lack of interpretability**. To ensure **transparency**, we implement:

### **1. Grad-CAM (Gradient-weighted Class Activation Mapping)**
- Highlights **important regions** in fundus images that influence the model’s decision.

### **2. Attention Maps**
- Focuses on **important spatial regions** in the image.

### **Example of Grad-CAM Output**
![Grad-CAM Example](gradcam.jpg)

---

## 🎓 **Training Strategy**
### **1. Loss Function**
- **Focal Loss with Label Smoothing** (`α=0.25, γ=2.0`).

### **2. Optimizer**
- **AdamW (Weight Decay Optimization)**.

### **3. Learning Rate Scheduling**
- Reduce LR if no improvement (`ReduceLROnPlateau`).

### **4. Early Stopping**
- Stops training if validation loss **does not improve for 10 epochs**.

---

## 📈 **Evaluation Metrics**
We use **multiple evaluation metrics**:

1. **Accuracy**
2. **ROC-AUC Score**
3. **Class Membership Probabilities**

### **Test Performance**
- **ROC-AUC Score = 0.9711**
- **Test Accuracy = 89.7%**

---

## 🏆 **Results**
| Class | No DR | Mild/Moderate DR | Severe/Proliferative DR |
|-------|-------|----------------|---------------------|
| **Probability** | 0.5016 | 0.3255 | 0.1730 |

---

## ⚙ **Installation**
### **1️⃣ Clone Repository**
```bash
git clone https://github.com/shekharkshitij/Diabetic_Retinopathy_Detection_with_Fuzzy_Logic.git

---

## 🔮 **Future Work**
- ✅ Implement more explainability techniques such as **Layer-wise Relevance Propagation (LRP)** and **SHAP values**.
- ✅ Improve the **Fuzzy Membership Layer** to enhance **uncertainty quantification**.
- ✅ Expand the model to detect other **eye diseases** such as **Glaucoma** and **Age-related Macular Degeneration (AMD)**.
- ✅ Deploy the model as a **Web Application** using **Streamlit** for real-time inference and explainability.

---

## 📚 **References**
1. [APTOS 2019 Blindness Detection Dataset - Kaggle](https://www.kaggle.com/c/aptos2019-blindness-detection/data)  
2. **EfficientNetV2: Smaller Models and Faster Training** - [Tan & Le, 2021](https://arxiv.org/abs/2104.00298)  
3. **Grad-CAM: Visual Explanations for Convolutional Neural Networks** - [ICML 2017](https://arxiv.org/abs/1610.02391)  

---

## ⭐ **Support**
If you find this project useful, please consider **starring ⭐ the repository** to show your support!
