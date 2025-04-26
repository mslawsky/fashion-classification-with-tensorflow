# 👕 Fashion MNIST Classification with TensorFlow

[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![Python](https://img.shields.io/badge/Python-3.6+-blue.svg)](https://www.python.org/)
[![Keras](https://img.shields.io/badge/Keras-2.x-red.svg)](https://keras.io/)
[![NumPy](https://img.shields.io/badge/NumPy-1.19+-green.svg)](https://numpy.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-3.3+-yellow.svg)](https://matplotlib.org/)

A neural network implementation using TensorFlow to classify fashion items from the Fashion MNIST dataset. This project demonstrates image classification fundamentals including data preprocessing, model building, training, and evaluation.

![Fashion MNIST Examples](https://github.com/zalandoresearch/fashion-mnist/raw/master/doc/img/fashion-mnist-sprite.png)

---

## 📋 Table of Contents
- [Project Overview](#-project-overview)
- [Dataset Details](#-dataset-details)
- [Model Architecture](#-model-architecture)
- [Training Process](#-training-process)
- [Results](#-results)
- [Installation & Usage](#-installation--usage)
- [Exploration Exercises](#-exploration-exercises)
- [Key Learnings](#-key-learnings)
- [Future Improvements](#-future-improvements)

---

## 🔎 Project Overview

This project builds a neural network model to recognize and classify clothing items from grayscale images. Unlike traditional "Hello World" examples that learn simple linear relationships, this project tackles a more challenging computer vision problem that showcases the power of neural networks in image recognition tasks.

**Key Objectives:**
- Load and preprocess the Fashion MNIST dataset
- Build and train a neural network classification model
- Visualize and understand the training process
- Evaluate model performance on unseen data
- Experiment with different model architectures and parameters

---

## 📊 Dataset Details

The Fashion MNIST dataset includes 70,000 grayscale images of clothing items (28x28 pixels):
- 60,000 training images
- 10,000 test images

Each image is labeled with one of 10 clothing categories:

| Label | Description | Example |
|-------|-------------|---------|
| 0 | T-shirt/top | ![T-shirt](https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/doc/img/fashion-mnist-sprite.png) |
| 1 | Trouser | ![Trouser](https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/doc/img/fashion-mnist-sprite.png) |
| 2 | Pullover | ![Pullover](https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/doc/img/fashion-mnist-sprite.png) |
| 3 | Dress | ![Dress](https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/doc/img/fashion-mnist-sprite.png) |
| 4 | Coat | ![Coat](https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/doc/img/fashion-mnist-sprite.png) |
| 5 | Sandal | ![Sandal](https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/doc/img/fashion-mnist-sprite.png) |
| 6 | Shirt | ![Shirt](https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/doc/img/fashion-mnist-sprite.png) |
| 7 | Sneaker | ![Sneaker](https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/doc/img/fashion-mnist-sprite.png) |
| 8 | Bag | ![Bag](https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/doc/img/fashion-mnist-sprite.png) |
| 9 | Ankle boot | ![Ankle boot](https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/doc/img/fashion-mnist-sprite.png) |

**Data Preprocessing:**
- Images are normalized from 0-255 pixel values to 0-1 range
- Labels are represented as integers from 0-9

---

## 🧠 Model Architecture

The neural network uses a straightforward architecture optimized for image classification:

```python
model = tf.keras.models.Sequential([
    tf.keras.Input(shape=(28,28)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation=tf.nn.relu),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])

model.compile(optimizer=tf.optimizers.Adam(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
```

**Architecture Breakdown:**
- **Input Layer**: Accepts 28x28 grayscale images
- **Flatten Layer**: Converts 2D image arrays (28x28) to 1D arrays (784)
- **Hidden Layer**: 128 neurons with ReLU activation
- **Output Layer**: 10 neurons (one per clothing category) with Softmax activation
- **Optimizer**: Adam (adaptive learning rate)
- **Loss Function**: Sparse Categorical Crossentropy

This architecture strikes a balance between simplicity and effectiveness for this classification task.

---

## 🔄 Training Process

The model is trained for 5 epochs using the prepared dataset:

```python
# Train the model
history = model.fit(training_images, training_labels, epochs=5)

# Evaluate on test data
test_loss, test_accuracy = model.evaluate(test_images, test_labels)
print(f"Test accuracy: {test_accuracy:.4f}")
```

**Training Visualization:**

![Training Accuracy Curve](https://i.imgur.com/YLdxNZN.png)

The graph shows steady improvement in accuracy across the training epochs, with the model quickly learning to distinguish between different clothing items.

---

## 📈 Results

After training for just 5 epochs, the model achieves impressive results:

| Metric | Training Set | Test Set |
|--------|--------------|----------|
| Accuracy | ~83% | ~82% |
| Loss | ~0.48 | ~0.50 |

**Classification Visualization:**

For an ankle boot image (label 9), the model outputs probability scores:
```
[1.0767830e-06 1.8923657e-07 9.3867056e-06 1.4331826e-05 3.1927171e-05
 1.6217418e-01 1.6793387e-05 2.9690662e-01 4.1863704e-03 5.3665912e-01]
```

The highest probability (0.536) correctly corresponds to class 9 (ankle boot).

![Prediction Example](https://i.imgur.com/JGubzO8.png)

---

## 🚀 Installation & Usage

### Prerequisites
- Python 3.6+
- TensorFlow 2.x
- NumPy
- Matplotlib

### Setup
```bash
# Clone this repository
git clone https://github.com/yourusername/fashion-mnist-classification.git

# Navigate to the project directory
cd fashion-mnist-classification

# Install dependencies
pip install tensorflow numpy matplotlib
```

### Running the Notebook
```bash
jupyter notebook C1_W2_Lab_1_beyond_hello_world.ipynb
```

### Example Code
```python
# Load the Fashion MNIST dataset
fmnist = tf.keras.datasets.fashion_mnist
(training_images, training_labels), (test_images, test_labels) = fmnist.load_data()

# Normalize the images
training_images = training_images / 255.0
test_images = test_images / 255.0

# Build the model
model = tf.keras.models.Sequential([
    tf.keras.Input(shape=(28,28)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation=tf.nn.relu),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])

# Compile the model
model.compile(optimizer=tf.optimizers.Adam(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(training_images, training_labels, epochs=5)

# Make predictions
predictions = model.predict(test_images)
```

---

## 🧪 Exploration Exercises

The notebook includes several exercises to deepen your understanding:

1. **Neuron Count Experiments**: Test different numbers of neurons in the hidden layer
   - Results show that increasing from 128 to 512 neurons improves accuracy but increases training time

2. **Layer Structure**: Explore the impact of adding or removing layers
   - Adding a second hidden layer can capture more complex patterns but may require more training time

3. **Training Duration**: Analyze the effect of training for more or fewer epochs
   - Training beyond 5-10 epochs shows diminishing returns and potential overfitting

4. **Early Stopping**: Implement callbacks to stop training when desired accuracy is reached
   ```python
   class myCallback(tf.keras.callbacks.Callback):
       def on_epoch_end(self, epoch, logs={}):
           if(logs.get('accuracy') >= 0.85):
               print("\nReached 85% accuracy - stopping training!")
               self.model.stop_training = True
   ```

---

## 🔍 Key Learnings

This project demonstrates several essential concepts in neural network development:

1. **Image Preprocessing**: Normalizing pixel values for optimal training
2. **Activation Functions**: Using ReLU for hidden layers and Softmax for multi-class output
3. **Model Evaluation**: Distinguishing between training and test performance
4. **Overfitting**: Recognizing when a model performs better on training than test data
5. **TensorFlow/Keras API**: Working with Sequential models and configuring training

---

## 🔮 Future Improvements

While this model achieves good accuracy, several enhancements could further improve performance:

1. **Convolutional Layers**: Add CNN layers specifically designed for image processing
2. **Data Augmentation**: Generate additional training examples through image transformations
3. **Regularization**: Implement dropout or L2 regularization to prevent overfitting
4. **Hyperparameter Tuning**: Systematically search for optimal learning rates and network sizes
5. **Transfer Learning**: Apply pre-trained models to achieve even higher accuracy

---

## 📫 Contact

For inquiries about this project:
- [GitHub Profile](https://github.com/yourusername)
- [LinkedIn](https://www.linkedin.com/in/yourprofile/)
- [Email](mailto:your.email@example.com)

---

© 2025 Your Name. All Rights Reserved.
