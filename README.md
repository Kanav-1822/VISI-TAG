# Enhanced Attribute Classifier

## 📌 Overview
The **Enhanced Attribute Classifier** is a deep learning model designed for **multi-attribute classification of images**. Built from scratch, it leverages a **ResNet-18** backbone for feature extraction, an **attention mechanism** for weighted feature refinement, and a **category-aware embedding layer** to enhance classification accuracy. This model is designed to handle large-scale datasets efficiently while ensuring robust performance across multiple attributes.

## 🚀 Key Features
- **Deep Learning Model from Scratch** – Designed and implemented a multi-attribute classifier without relying on pre-built solutions.
- **ResNet-18 Backbone** – Uses a pre-trained ResNet-18 network as a **feature extractor**, replacing the classification head with a custom multi-attribute prediction architecture.
- **Attention Mechanism** – Integrates a **self-learned attention module** to refine feature representations and improve classification accuracy.
- **Category-Aware Embeddings** – Incorporates an **embedding layer** that learns category-specific information to enhance predictions.
- **Handles Large Datasets** – Optimized for high-dimensional image datasets, leveraging efficient batch processing, dropout, and normalization.
- **Multi-Task Learning** – Predicts multiple attributes per image by utilizing separate output heads for each attribute.

## 🏗 Model Architecture
### 1️⃣ **Feature Extraction**
- ResNet-18 acts as a backbone, where the final classification layer is replaced with an **Identity Layer**.
- Extracted deep feature vectors are then fed into the **attention module**.

### 2️⃣ **Attention Mechanism**
- The **attention layer** learns the importance of extracted features.
- It consists of **fully connected layers with ReLU activation**, ultimately generating a weight for each feature.
- The weighted features improve the model's focus on the most relevant regions of the image.

### 3️⃣ **Category Embeddings**
- A **trainable embedding layer** encodes category-specific information.
- Each image’s category ID is converted into a dense **50-dimensional** embedding.
- The embedding is concatenated with extracted features to improve classification.

### 4️⃣ **Attribute Prediction Heads**
- A **ModuleList of dense layers**, where each head predicts a separate attribute.
- Uses **Batch Normalization, ReLU activation, and Dropout** to improve generalization.
- Outputs **sigmoid activations** for binary attribute classification.

## 📊 Training and Optimization
- **Dataset Handling**: Processed a large dataset with multiple attributes per image.
- **Loss Function**: Binary Cross Entropy (BCE) loss for independent attribute predictions.
- **Optimizer**: AdamW optimizer with adaptive learning rates.
- **Regularization**: Applied **Dropout (0.5)** and **Batch Normalization** to prevent overfitting.
- **Efficient Training**: Utilized **gradient accumulation** and multi-GPU processing for scalability.

## 🖥️ Installation & Usage
### 🔧 Prerequisites
Ensure you have Python 3.8+ installed and the following dependencies:
```bash
pip install torch torchvision numpy pandas scikit-learn
```

### 🏃‍♂️ Run the Model
```python
import torch
from model import EnhancedAttributeClassifier

num_attributes = 77  
num_categories = len(subset_train_data['Category'].unique())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

model = EnhancedAttributeClassifier(num_attributes, num_categories).to(device)
print(model)
```

## 📈 Results & Performance
- Achieved **high classification accuracy** across multiple attributes.
- Attention mechanism improved focus on key image regions.
- Category embeddings significantly enhanced model performance.

