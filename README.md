# IMAGE-CLASSIFICATION-MODEL

# DECISION-TREE-IMPLEMENTATION

*COMPANY*:CODETECH IT SOLUTIONS

*NAME*:RAVAVARAPU VENKATA VARALAKSHMI DEVI

*INTERN ID*:CT06DL1153

*DOMAIN*:MACHINE LEARNING

*DURATION*:6 WEEKS

*MENTOR*:NEELA SANTOSH

DESCRIPTION:
This Python script demonstrates a complete image classification workflow using a Convolutional Neural Network (CNN) on the CIFAR-10 dataset with TensorFlow and Keras. It begins by importing necessary libraries such as NumPy, Matplotlib, Seaborn, OpenCV, and scikit-learn. The CIFAR-10 dataset is loaded and normalized to scale the pixel values between 0 and 1. Class labels are defined for each of the ten categories (e.g., airplane, dog, truck), and a grid of 100 sample training images is visualized for an initial understanding of the data. The CNN architecture consists of three convolutional layers with ReLU activation, followed by max-pooling layers to reduce spatial dimensions. A flattening layer is then used to convert the output into a one-dimensional array, which is passed through a dense hidden layer with 128 neurons and a dropout layer to prevent overfitting. The final output layer uses softmax activation to perform multi-class classification. The model is compiled using the Adam optimizer and trained with the sparse categorical crossentropy loss function for 10 epochs, using a batch size of 64.

After training, the model is evaluated on the test set to determine its accuracy. Predictions are made on the test images, and a confusion matrix is generated using Seaborn to visualize the performance across all categories. Additionally, a classification report is printed to show precision, recall, and F1-score for each class. The trained model is saved in the Keras format for later use. The script also includes a custom function, `predict_external_image`, which allows prediction on an external image. This function reads an image file using OpenCV, converts it to RGB format, resizes it to 32x32 pixels to match the model’s input size, normalizes it, and reshapes it for prediction. The model then predicts the class and confidence level, displaying the image with the predicted label using Matplotlib. This feature showcases how the model can be deployed to classify real-world images, making the script a practical tool for both learning and implementation of CNNs for image classification tasks.

#OUTPUT

![Image](https://github.com/user-attachments/assets/c0cd87c4-fd85-4c68-9348-ae81fd32457c)

![Image](https://github.com/user-attachments/assets/f7f4007a-7410-4076-8da5-33c5472242a3)

![Image](https://github.com/user-attachments/assets/9a1d4647-4eeb-47e9-9cc5-746ba2d2023c)

![Image](https://github.com/user-attachments/assets/5a438350-42e1-4bf9-8bf5-1f2e9649d0c6)

![Image](https://github.com/user-attachments/assets/e1cc9b10-fba3-44bd-af50-de12ba3b8eff)
