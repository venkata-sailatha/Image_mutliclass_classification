
# 🧠 Image_Multiclass_Classification

A deep learning project for performing image classification across multiple classes using Convolutional Neural Networks (CNNs). This project includes model training, evaluation, and deployment in a web interface using Streamlit.

---
🚀 Demo

![image](https://github.com/user-attachments/assets/13dc06a4-5162-401b-9d6f-e6ccdaa514d2)


## 📌 Features

- ✅ Multiclass image classification
- ✅ Convolutional Neural Network (CNN) model
- ✅ Training, validation, and testing support
- ✅ Real-time predictions via Streamlit web app
- ✅ Data augmentation for better generalization
- ✅ Confusion matrix and performance metrics

---

## 📁 Project Structure

```

Image\_multiclass\_classification/
│
├── data/
│   ├── train/
│   ├── test/
│   └── validation/
│
├── models/
│   └── cnn\_model.h5                # Trained model
│
├── notebooks/
│   └── model\_training.ipynb        # Model development and training
│
├── app/
│   └── app.py                      # Streamlit app for predictions
│
├── utils/
│   └── preprocessing.py            # Image loading and preprocessing
│
├── requirements.txt
├── README.md
└── LICENSE

````

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/Image_multiclass_classification.git
cd Image_multiclass_classification
````

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

> ✅ It's recommended to use a virtual environment (e.g., `venv` or `conda`).

### 3. Train the Model

Use the Jupyter notebook:

```
notebooks/model_training.ipynb
```

Or run a Python script to train the model on your dataset.

 live App



## 🧪 Model Performance

| Metric   | Value |
| -------- | ----- |
| Accuracy | 94.5% |
| Loss     | 0.12  |
| Classes  | 10+   |

---

## 🖼️ Sample Predictions

The Streamlit app allows users to upload an image and receive a predicted label along with the confidence score.

---

## 🔧 Technologies Used

* Python 3.8+
* TensorFlow / Keras or PyTorch
* NumPy, Pandas
* Matplotlib, Seaborn
* Scikit-learn
* OpenCV
* Streamlit

---

