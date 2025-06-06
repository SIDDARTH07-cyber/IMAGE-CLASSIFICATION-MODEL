# IMAGE-CLASSIFICATION-MODEL

COMPANY: CODTECH IT SOLUTIONS

NAME: P VISHNU SIDDARTH

INTERN ID: CT04DF2078

DURATION: 4 WEEKS

MENTOR: NEELA SANTOSH

📌 Project Overview
This project demonstrates the development and evaluation of an Image Classification Model using a Convolutional Neural Network (CNN) built with TensorFlow and Keras. The model is trained on the popular CIFAR-10 dataset, which contains 60,000 color images categorized into 10 different classes, such as airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck.

The goal is to train a CNN model that can accurately classify unseen test images into one of these ten categories, showcasing a foundational yet powerful application of deep learning in computer vision. This project fulfills the requirements of the internship deliverable for CodTech, and includes all key components: data preprocessing, model construction, training, evaluation, and model saving.

🚀 Objectives
Build a functional deep learning model for image classification.

Train the model using a real-world dataset (CIFAR-10).

Evaluate the model's performance on a separate test dataset.

Visualize the accuracy trends over training epochs.

Save the final trained model for reuse or deployment.

🧰 Tools & Technologies Used
Language: Python

Framework: TensorFlow (Keras API)

Dataset: CIFAR-10 (available via tf.keras.datasets)

Libraries: TensorFlow, NumPy, Matplotlib

🧠 Model Architecture
The model is built using the Sequential API from Keras and consists of the following layers:

Convolutional Layer 1: 32 filters, 3x3 kernel, ReLU activation

MaxPooling Layer 1: 2x2 pool size

Convolutional Layer 2: 64 filters, 3x3 kernel, ReLU activation

MaxPooling Layer 2: 2x2 pool size

Convolutional Layer 3: 64 filters, 3x3 kernel, ReLU activation

Flatten Layer

Dense Layer: 64 neurons, ReLU activation

Output Layer: 10 neurons, softmax activation (for multi-class classification)

This architecture is designed to capture spatial hierarchies in the image data and perform robust classification over the CIFAR-10 categories.

📈 Model Training
The model is compiled using the Adam optimizer and trained with sparse categorical crossentropy as the loss function. It is trained for 10 epochs with both training and validation accuracy being tracked. CIFAR-10 data is normalized by dividing pixel values by 255 to speed up convergence.

After training, the model is evaluated on the test set and achieves a strong level of accuracy, which confirms its ability to generalize on unseen data.

✅ Model Evaluation
The final test accuracy is printed after model evaluation, and a visualization of training vs validation accuracy is plotted using Matplotlib. This gives insight into the model’s learning progress and any potential overfitting.

💾 Model Export
After training and evaluation, the model is saved as an .h5 file (cnn_model_codtech.h5) so it can be reloaded or deployed in the future without retraining.

📂 Directory Structure
cpp
Copy
Edit
📦 CODE-TECH/
 ┣ 📄 image_classification.py
 ┣ 📄 cnn_model_codtech.h5
 ┗ 📊 output plots (optional)
🧪 How to Run the Project
Install TensorFlow:

pip install tensorflow

Run the script:

python image_classification.py
View the printed test accuracy and training plot.

📌 Future Improvements
Implement data augmentation to reduce overfitting.

Use a deeper CNN architecture or try ResNet for better performance.

Add a confusion matrix and classification report for detailed analysis.

Deploy the model as a web app using Flask or Streamlit.

🏁 Conclusion
This project highlights the use of CNNs in solving image classification tasks using TensorFlow. It successfully builds a model that learns to classify images across 10 categories and provides a strong foundation for exploring more advanced computer vision projects. It fulfills the CodTech internship task of building and evaluating a functional CNN model with a clear deliverable.

#OUTPUT

![Image](https://github.com/user-attachments/assets/d6ee1c43-a5ce-412d-b898-beebc98e4c50)

![Image](https://github.com/user-attachments/assets/1537b661-45ed-4acb-87de-43f41552967c)

![Image](https://github.com/user-attachments/assets/a424c38b-4741-46f0-b53e-349b3d138c6e)

