# Cat vs Dog Image Classifier 🐱🐶

This project implements a Convolutional Neural Network (CNN) to classify images of cats and dogs. Built using deep learning libraries such as TensorFlow or PyTorch, the model is trained on labeled image data and can accurately distinguish between the two categories. The project includes steps for data preprocessing, model training, evaluation, and making predictions on new images.

## 📂 Dataset Preparation

To train the model, you must manually create two folders inside a directory named `dataset/`:

dataset/
├── cats/
│ ├── cat1.jpg
│ ├── cat2.jpg
│ └── ...
└── dogs/
├── dog1.jpg
├── dog2.jpg
└── ...


Each folder (`cats/` and `dogs/`) should contain a large number of labeled images representing each class.

> ⚠️ **Note:** Due to GitHub's file upload limits (100 files per directory), the image dataset is not included in this repository. You will need to use your own dataset or download a suitable one (e.g., from Kaggle).

## ✅ Features

- Image preprocessing with data augmentation
- CNN architecture for binary classification
- Model training and validation
- Accuracy evaluation
- Prediction on custom images

## 🛠️ Technologies Used

- Python
- TensorFlow / Keras 
- NumPy
