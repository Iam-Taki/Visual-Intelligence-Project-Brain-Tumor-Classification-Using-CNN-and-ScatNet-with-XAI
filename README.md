# Visual-Intelligence-Project-Brain-Tumor-Classification-Using-CNN-and-ScatNet-with-XA
🧠 Brain Tumor Detection Using CNN and ScatNet with Explainable AI








📌 Project Description

This project presents a comparative study of Convolutional Neural Networks (CNNs) and Scattering Networks (ScatNet) for binary brain tumor classification using MRI images.
In addition to performance evaluation, the project strongly emphasizes model transparency and interpretability through multiple Explainable Artificial Intelligence (XAI) techniques.

Both models are evaluated under a fair experimental setup:

identical classifier architecture

5-fold cross-validation

consistent preprocessing

standardized evaluation metrics

🎓 Academic Context

Course: Visual Intelligence

Degree: M.Sc. in Artificial Intelligence

University: University of Verona (Italy)

Project Type: Medical Image Classification + Explainable AI

🎯 Objectives

Implement and compare CNN and ScatNet for brain tumor detection.

Ensure a fair comparison using the same classifier for both models.

Perform 5-fold cross-validation.

Evaluate performance using Accuracy, F1-score, Precision, Recall.

Extract and visualize CNN learned filters and ScatNet wavelet filters.

Apply multiple XAI methods to interpret model predictions.

Implement one XAI method from scratch and validate it using Captum.

Compare interpretability between CNN and ScatNet.

🧬 Dataset

Type: Brain MRI images

Task: Binary classification (Tumor / No Tumor)

Preprocessing

Image resizing

Intensity normalization

Optional data augmentation

⚠️ Note:
The dataset path must be updated in the notebook before execution.

🏗️ Model Architectures
🔹 Convolutional Neural Network (CNN)

Multiple convolutional layers

ReLU activations

Max-Pooling & Dropout

Fully connected classifier

Learned, adaptive filters

🔹 Scattering Network (ScatNet)

Wavelet-based feature extraction (Kymatio)

Fixed, mathematically defined filters

Same fully connected classifier as CNN

Stable but less adaptive representations

📊 Evaluation Protocol

Validation: 5-Fold Cross-Validation

Metrics:

Accuracy

F1-Score

Precision

Recall

Visualizations

Training & validation learning curves

Confusion matrices

CNN vs ScatNet filter visualizations

🔍 Explainable AI (XAI)
✔ Implemented XAI Methods

DeepLIFT

SHAP

Occlusion

Gradient-based attribution methods

✔ Custom XAI

One XAI method implemented from scratch

Compared with Captum implementation for validation

✔ XAI Analysis

Attribution maps overlaid on MRI images

At least two images per class

CNN vs ScatNet qualitative comparison

📈 Key Results

CNN achieved higher accuracy and F1-score than ScatNet.

CNN filters adapted to tumor-specific features.

ScatNet provided stable but less discriminative representations.

CNN attribution maps were well-localized and clinically meaningful.

Captum results confirmed correctness of manual XAI implementations.

🛠️ Technologies Used

Python

PyTorch

Kymatio

Captum

NumPy, Matplotlib, Scikit-learn

▶️ How to Run
1️⃣ Install dependencies
pip install torch torchvision captum kymatio scikit-learn matplotlib

2️⃣ Prepare dataset

Place the dataset in your local or Google Drive directory

Update dataset paths inside the notebook

3️⃣ Run the notebook
jupyter notebook


Execute cells sequentially to:

Train CNN & ScatNet

Perform cross-validation

Generate metrics

Visualize filters & XAI attributions

📌 Conclusion

CNNs outperform ScatNet for brain tumor detection in MRI images, achieving better classification performance and more interpretable predictions. While ScatNet offers theoretical stability through wavelet-based features, CNNs provide greater adaptability and more meaningful explanations, making them more suitable for explainable medical diagnosis.

🔮 Future Work

Hybrid CNN–ScatNet architectures

Multi-class tumor classification

3D MRI volume analysis

Clinical validation with expert annotations

👤 Author

Abdullah Al Noman Taki
M.Sc. Artificial Intelligence
University of Verona

⭐ If you find this project useful, feel free to star the repository!
