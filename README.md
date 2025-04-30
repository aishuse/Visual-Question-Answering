**Visual Question Answering (VQA) - Abstract Scenes**

This project implements a Visual Question Answering (VQA) system using the VQA Abstract Scenes Dataset. It takes a cartoon-style image and a natural language question about the image, and predicts an answer.

🔍 **Dataset**

Name: VQA Abstract Scene Dataset (v2)

Source: VQA Dataset

The dataset contains synthetic scene images built from clipart objects, designed to test reasoning over structured visual scenes.

🧠 **Model Architecture**

The model uses a combination of:

Image Encoder: Frozen ResNet50 with Global Average Pooling and a Dense Layer to extract image features (input size 224x224).

LSTM-based Question Encoder: Pretrained word embeddings are passed into stacked LSTMs to create a Question Context Vector.

Cross-modal Attention: Multi-Head Cross-Modal Attention is used to align image and question representations.

Fusion Layer: Combines attended image features, global image features, and the question summary.

Classifier: Fully-connected Dense layers (512 → 256) followed by a Softmax over 1,000 answer classes.

The model is trained using the Categorical Crossentropy loss and the Adam optimizer.
Checkpointing is used to save model weights after each epoch.

⚙️ **Hyperparameters**

Loss Function: Categorical Crossentropy

Optimizer: Adam

Evaluation Metric: Accuracy

Batch Size: 128 (initial), 32 (for fine-tuning)

Learning Rate: 0.00001

📊 **Best Model Performance**

Best Epoch: 24 (out of 30 total)

**Exact Match: 27.55%**

Partial Match: 41.21%

**Validation Accuracy: 51.8%**

Validation Loss: 1.7154

Training Accuracy: 54.76%

Training Loss: 1.5631

