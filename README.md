# EmoSense: Emotion Detection using Convolutional Neural Networks (CNN)

Emotion Detection and Speech Therapy App for People with Autism Spectrum Disorder (ASD).

## Team Information
**Team ID:** C241-PS141

| Member ID      | Name                     | Institution                  |
|----------------|--------------------------|------------------------------|
| M001D4KX1840   | Cheisha Amanda           | Institut Pertanian Bogor     |
| M012D4KX2134   | Sofi Nurhayati Latifah   | Universitas Telkom           |
| M012D4KX2314   | Nabila Aurellia          | Universitas Telkom           |
| C010D4KX1282   | Dhina Rotua Mutiara      | Universitas Indonesia        |
| C010D4KX1305   | Mayfa Shadrina Siddi     | Universitas Indonesia        |
| A010D4KX3539   | Refiany Shadrina         | Universitas Indonesia        |
| A012D4KX4060   | Aisha Farizka Mawla      | Universitas Telkom           |

## Project Overview

EmoSense is an application that provides a place for people with ASD to practice their emotional interpretation and some additional features such as a community platform for sharing experiences and support, health and community articles, and recommendations regarding clinic/therapy locations. This application targets the family members of people with ASD, hence the main users will be them. This application is expected to be a safe place for the family members to engage with their children/siblings who live with ASD more. In this project, we use Convolutional Neural Networks (CNNs) to classify facial expressions into seven different emotions: anger, disgust, fear, happiness, sadness, surprise, and neutral.

## Dataset

The data consists of images depicting 7 types of facial expressions, taken from [OSF](https://osf.io/f7zbv/). There are approximately a total of 700 images, divided in a ratio of 70:20:10. About 490 images have been allocated to the training set, 140 to the validation set, and the remaining 70 to the test set.

## Training the Emotion Detection Model

### 1. Dataset Preparation:
   - **Download and Extraction**: Download the dataset from Google Drive using its file ID and extract it into a designated directory (`/content/facerecog/EmoSense_Dataset2`). Ensure all required image data for training, validation, and testing are properly organized within their respective folders (`train`, `valid`, `test`).

### 2. Data Preprocessing and Augmentation:
   - **ImageDataGenerator**: Implement data augmentation techniques such as rotation, shifting, shearing, zooming, and flipping using TensorFlow's `ImageDataGenerator` class. This step is crucial for enhancing the model's ability to generalize on unseen data and prevent overfitting.

### 3. Model Architecture:
   - **Base Model Selection**: Utilize the VGG16 pre-trained model as the base architecture for feature extraction from input images. This model is chosen for its effectiveness in image recognition tasks and can be fine-tuned to suit specific emotion classification requirements.

   - **Custom Layers**: Add additional layers on top of the base model, including Global Average Pooling, Batch Normalization, Dropout, and Dense layers. These layers help in extracting relevant features from the input data and improve the model's performance.

### 4. Model Compilation:
   - **Compile the Model**: Configure the model with appropriate loss function (`categorical_crossentropy` for multi-class classification), optimizer (`Nadam` with a learning rate of `1e-4`), and evaluation metrics (`accuracy`). This step prepares the model for training using the specified parameters.

### 5. Model Training:
   - **Training Process**: Train the compiled model using the prepared data generators (`train_generator` for training data and `validation_generator` for validation data). Monitor training progress using callbacks such as Early Stopping to prevent overfitting and Model Checkpoint to save the best model based on validation accuracy.

   - **Epochs and Batch Size**: Specify the number of epochs (`50` in this case) and batch size (`16` images per batch) to iterate over the entire dataset multiple times while adjusting model weights based on computed gradients.

### 6. Model Evaluation:
   - **Evaluation Metrics**: Evaluate the trained model's performance on the test dataset (`test_generator`). Compute metrics such as loss and accuracy to assess how well the model generalizes to unseen data, providing insights into its effectiveness in emotion detection.

### 7. Model Conversion and Deployment:
   - **Convert to TensorFlow.js**: Convert the trained Keras model into TensorFlow.js format (`tfjs_model`) for web-based deployment. This step ensures compatibility and efficient execution of the model in a browser or any JavaScript environment.

### 8. Results and Visualization:
   - **Visualization**: Visualize training and validation metrics (accuracy and loss) using Matplotlib to analyze model performance across epochs. This visual feedback helps in understanding how well the model learns from training data and generalizes to validation data.

## Model Architecture

The CNN model architecture consists of multiple convolutional layers with batch normalization, max pooling, dropout for regularization, and dense layers for classification. The model is compiled with Adam optimizer and categorical crossentropy loss function.

## Files Structure

- `EmoSense.ipynb`: Jupyter notebook containing the entire project code including data preparation, model building, training, and evaluation.
- `requirements.txt`: List of Python packages required to run the project.
- `EmoSense_model.h5`: Trained CNN model saved in HDF5 format.
- `EmoSense_TFjs.zip`: Consist of file that converted by TensorFlow.js for web deployment.
- `EmoSense_Dataset.zip`: Data of this project.
- `README.md`: Project overview, setup instructions, and details.

## Results

The model achieves competitive accuracy and loss metrics on the validation set, demonstrating its effectiveness in emotion detection from facial expressions.

## Acknowledgments

Special thanks to Bangkit 2024 for providing the opportunity to work on this capstone project and to the instructors and mentors for their guidance.

---
