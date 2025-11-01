# 🧠 Brain Tumor Classification Using Convolutional Neural Networks (CNN)

This project applies **Deep Learning** techniques to detect and classify **brain tumors** from MRI images.  
Using a **Convolutional Neural Network (CNN)** built with **TensorFlow/Keras**, the model learns spatial and visual patterns that differentiate between **tumor** and **non-tumor** MRI scans — providing a step toward faster, more reliable medical diagnostics.

---

## 🗂️   Project Structure
```
brain_tumor_detection/
├── DATASET/            # Contains Training and Testing folders housing data for each of the brain tumor categories
├── models/             # The best model saved from training
├── notebooks/          # Jupyter notebook for analysis and modeling
├── outputs/            # Generated outputs like confusion matrix, accuracy and loss plots, model predictions
├── README.md           # Project overview and documentation
├── requirements.txt    # List of Python dependencies for easy setup
```

---

## 📊 Dataset Overview

**Source:** [Brain Tumor MRI Dataset on Kaggle](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset/data)

**Description:**
- MRI images of human brains, categorized into:
  - `glioma_tumor`
  - `meningioma_tumor`
  - `pituitary_tumor`
  - `no_tumor`
- Organized into:
  - `Training` and `Testing` directories in the DATASET folder.
  - Each directory contains class-based subfolders.
- Approximately 7,000+ MRI images in total.

---

## 🎯 Objective

The goal of this project is to:
1. Build a CNN model that accurately classifies MRI scans into their respective categories.
2. Evaluate its performance using various metrics.
3. Visualize and interpret the CNN’s learning process.

---

## ⚙️ Methodology

### 🧩 1. Data Loading & Visualization
- Created a function to load train and test datasets into a dataframe.
- Visualized random samples from each tumor category.
- Checked dataset structure, class balance, and file counts.

### 🧼 2. Data Preprocessing
- Converted all images to RGB and resized to a fixed shape (`150x150x3`).
- Normalized pixel values to the range `[0,1]`.
- Applied **data augmentation** (rotation, zoom, horizontal flip) to improve generalization.
- Split Training dataset into **training** and **validation** sets.

### 🧠 3. Model Development
Built a **custom CNN architecture** using Keras Sequential API:

| Layer Type | Description |
|-------------|-------------|
| Convolutional | Extracted visual features using filters |
| MaxPooling | Downsampled spatial dimensions |
| Dropout | Reduced overfitting |
| Flatten | Converted 2D features into 1D vector |
| Dense | Fully connected layers for decision-making |
| Output | `Softmax` activation for multi-class classification |

**Optimizer:** Adam  
**Loss Function:** Categorical Cross-Entropy  
**Evaluation Metric:** Accuracy

### 🏋️ 4. Model Training
- Trained the CNN over 50 epochs with early stopping implementation 
- Used **ImageDataGenerator** for real-time data feeding and augmentation.
- Monitored training and validation loss and accuracy.

### 🧾 5. Model Evaluation
- Evaluated model performance on the **test dataset**.
- Plotted **accuracy and loss curves** to observe convergence.
- Generated **confusion matrix** and **classification report** (precision, recall, F1-score).
- Visualized predictions with corresponding ground truths.

---

## 📈 Results

| Metric | Training | Validation |
|:--------|:----------:|:------------:|
| Accuracy | ~86% | ~79% |
| Loss | ~0.37 | ~0.60 |

### ✅ Key Observations:
- Model generalizes well across tumor types.
- Data augmentation significantly improved robustness.
- CNN effectively distinguishes subtle differences between tumor classes.

---

## 💡 Insights

- CNNs are powerful tools for medical imaging — they learn key spatial and textural features without manual feature extraction.
- **Glioma** and **Meningioma** tumors showed overlapping features, challenging classification — future improvements could leverage deeper networks or transfer learning.
- The model can assist radiologists by providing **AI-driven second opinions** in MRI analysis.

---

## 🚀 How to Run the Project

### Clone the Repository

```bash
git clone https://github.com/gabbyomekz/brain_tumor_detection.git
cd brain_tumor-detection
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run the Notebook

```bash
jupyter notebook CNN_for_brain_tumor_detection.ipynb
```

---

## 🧰 Tech Stack

* **Language:** Python
* **Framework:** TensorFlow / Keras
* **Visualization:** Matplotlib, Seaborn
* **Data Handling:** NumPy, Pandas
* **Image Processing:** OpenCV, Pillow
* **Tools:** Jupyter Notebook

---

## 💡 Future Improvements

* Apply Transfer Learning using pre-trained models (VGG16, ResNet50, EfficientNet).
* Implement **Grad-CAM** for visual interpretability.
* Deploy using **Streamlit** or **Flask** for interactive predictions.

---

## 👨<200d>💻 Author

Developed by [Gabriel Omeke]
📧 Contact: [gabrielomeke92@gmail.com](mailto:gabrielomeke92@gmail.com)
🔗 GitHub: [github.com/gabbyomekz](https://github.com/gabbyomekz)
