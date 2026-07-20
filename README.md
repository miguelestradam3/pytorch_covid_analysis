# 🩺 COVID-19 Detection from Chest X-ray Images using PyTorch

This project demonstrates how to build a deep learning model capable of classifying chest X-ray images into **three categories**:

- 🫁 Normal
- 🤒 Viral Pneumonia
- 🦠 COVID-19

Using **PyTorch** and **Transfer Learning** with a pretrained **ResNet-18** model, this notebook walks through the complete image classification pipeline, from data preparation to model evaluation.

---

## 📌 Features

- Chest X-ray image classification
- Multi-class classification (3 classes)
- Transfer Learning with ResNet-18
- Data augmentation
- Custom PyTorch Dataset
- Image preprocessing and normalization
- Model training and validation
- Prediction visualization
- Performance evaluation

---

## 🛠 Technologies Used

- Python
- PyTorch
- TorchVision
- NumPy
- Matplotlib
- Pillow (PIL)
- Jupyter Notebook

---

## 📚 Libraries

```python
torch
torchvision
numpy
matplotlib
Pillow
```

---

## 📊 Dataset

The project uses the **COVID-19 Radiography Dataset**, which contains chest X-ray images belonging to three different classes:

- Normal
- Viral Pneumonia
- COVID-19

The notebook automatically prepares training and testing datasets before training the model.

---

## 🔍 Data Preprocessing

Before training, the images undergo several preprocessing steps:

- Resize images to **224 × 224**
- Random horizontal flipping (training only)
- Convert images to tensors
- Normalize pixel values using ImageNet statistics

A custom PyTorch `Dataset` class is implemented to efficiently load and manage the images.

---

## 🤖 Deep Learning Model

The notebook uses **Transfer Learning** with a pretrained **ResNet-18** convolutional neural network.

Workflow:

1. Load chest X-ray images
2. Apply image transformations
3. Create DataLoaders
4. Load pretrained ResNet-18
5. Fine-tune the classifier
6. Train the model
7. Evaluate predictions

---

## 📈 Model Evaluation

The project includes:

- Training and validation
- Prediction on unseen images
- Visualization of predictions
- Accuracy monitoring during training

---

## 🚀 Getting Started

### Install dependencies

```bash
pip install -r requirements.txt
```

Or install manually:

```bash
pip install torch torchvision numpy matplotlib pillow
```

---

## ▶️ Running the Project

Launch Jupyter Notebook:

```bash
jupyter notebook
```

Open the jupyter notebook.

Run the notebook cells sequentially to reproduce the complete deep learning workflow.

---

## 🎯 Learning Objectives

This project demonstrates how to:

- Build image classification models with PyTorch
- Work with medical imaging datasets
- Apply transfer learning
- Create custom PyTorch datasets
- Perform image preprocessing and augmentation
- Train convolutional neural networks
- Visualize predictions and model performance

