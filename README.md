# cats-vs-dogs-classifier

The primary aim of this project is to develop a machine learning-based classifier capable of discerning between images of cats and dogs. The classifier should be robust enough to correctly categorize these images, despite variations in pose, thereby providing a solution for a binary classification problem.

### B. Objective of the assignment
The underlying goal of this assignment is to design, train, and evaluate a machine learning model that can reliably identify and classify images as either containing a cat or a dog. By harnessing the power of the given dataset, the intention is to create a model that can generalize well and predict the correct animal in unseen images, demonstrating the successful application of machine learning principles in a real-world context.

### C. Dataset Overview
The 'Dogs VS. Cats' dataset obtained from Kaggle was used for this project. The dataset is divided into two folders called test1 and train; for this task, the train dataset was used because it contained the appropriate image labels. There are 25,000 images of dogs and cats in the original train dataset, but the train folder with 1,004 images was used.

The images in the dataset show significant variation in pose, mirroring real-world conditions and increasing the problem's complexity and relevance. The original dataset is available at this link: [https://www.kaggle.com/c/dogs-vs-cats/data](https://www.kaggle.com/c/dogs-vs-cats/data).

The combination of a large, diverse dataset and inherent pose variation makes for an ideal testing environment for the development of an effective cat/dog image classifier.

## DATA PRE-PROCESSING

### A. Loading the Dataset
The original dataset for this project is divided into two folders: "test1" and "train," with a total of 37,500 images. However, not all of these photographs were used in the project. Instead, the "train" folder with 1004 images was used.

A custom Python function was written to load the dataset of cat and dog images. The function works by iterating through the JPEG images in the "train" folder path, with 1,004 loaded images. This method ensures that the dataset is handled efficiently.

The Image module of the Python Imaging Library (PIL) is used to load each image. After loading, the images are resized to a consistent size of 350 x 350 pixels and converted to NumPy arrays for further analysis and processing.

Each image's label was derived from its corresponding filename. The first part of the filename is used to determine whether an image depicts a cat or a dog. The images in the dataset were successfully loaded and prepared for the project's subsequent phases, allowing for the development and training of the classification model.


### B. Normalization of Images
Image normalization is a crucial preprocessing step in image processing when preparing data for machine learning models.

Normalization ensures that pixel intensity values are within a similar scale, allowing for better model performance and faster convergence. In this case, the pixel intensities were normalized to values between 0 and 1. To achieve this range, divide all pixel values by 255, which corresponds to the highest pixel intensity value in an 8-bit grayscale or RGB image.

By normalizing the pixel intensities, the influence of certain features dominating others is mitigated. This scaling ensures that the model can effectively learn from the normalized images, as the pixel values fall within a consistent range across different images.

### C. Label Conversion
Labels associated with the loaded images were converted from a categorical form to a numerical format.

In this specific case, the labels represented two classes: 'cat' and 'dog.' To enable machine learning models to process the labels effectively, they were converted to numeric values. The conversion assigned a value of 0 to the 'cat' class and a value of 1 to the 'dog' class.

By converting the labels to a numerical format, the models can interpret and analyze the labels during training. This numeric representation allows the models to learn the underlying patterns and relationships between the labels and the corresponding normalized images.

### D. Resizing of Images
During the dataset loading process, all images were resized to have the same dimensions (350 x 350). This step is essential as it ensures that all images fed into the machine learning model have the same number of features (pixels), which is a requirement for most machine learning algorithms. This was achieved using the PIL Image module's resize function.

### E. Dimension Reduction Using PCA
After normalization and resizing, dimension reduction was performed on the dataset using Principal Component Analysis (PCA). PCA is a dimensionality reduction technique that helps to reduce the complexity and size of the dataset by transforming the original variables into a new set of variables called Principal Components. These Principal Components are linear combinations of the original variables and are ordered such that the first few retain most of the variation present in all of the original variables. In this project, the number of Principal Components was set to 100.

### F. Data Splitting
The train_test_split function from the sklearn library was used to split the data. It partitioned the features (train_images_pca) and targets (train_labels) into training (x_train, y_train) and testing (x_test, y_test) sets, where 80% is used for training and 20% for testing. The random_state parameter ensures consistent splits across multiple runs for reproducibility. This division is crucial to train our model and to evaluate its ability to generalize on unseen data.

## MODEL TRAINING

### A. Algorithm Selection
The algorithms selected for evaluation are as follows:
- Support Vector Machine (SVM): A supervised learning algorithm known for its effectiveness in high-dimensional spaces.
- Random Forest Classifier: An ensemble learning method that constructs multiple decision trees and combines their predictions.
- XGBoost Classifier: A gradient boosting algorithm that has demonstrated high performance in various domains.

### B. Model Training
Each algorithm was trained on the preprocessed dataset using default hyperparameters, depending on the algorithm.

### C. Model Evaluation
The trained models were evaluated on a separate testing subset of the dataset to measure their performance. Evaluation metrics such as accuracy, precision, recall, and F1-score were calculated to assess the algorithms' classification capabilities. Additionally, the computational resources required by each algorithm during training and prediction were recorded.

## RESULTS AND DISCUSSION

### A. Performance Metrics Overview
The performance of each algorithm was assessed based on evaluation metrics including accuracy, precision, recall, and F1-score. The results are summarized in the following table:

| Algorithm        | Accuracy | Precision | Recall | F1-score |
|------------------|----------|-----------|--------|----------|
| SVM              | 0.597    | 0.570     | 0.670  | 0.616    |
| Random Forest    | 0.577    | 0.558     | 0.587  | 0.572    |
| XGBoost          | 0.621    | 0.617     | 0.567  | 0.591    |

Accuracy measures the proportion of total predictions that are correct. In this case, XGBoost achieved the highest accuracy at 62.1%, which suggests that it correctly identified the class of an image about 62.1% of the time. SVM and Random Forest followed with accuracy scores of approximately 59.7% and 57.7% respectively.

Precision is the ratio of true positives to the sum of true and false positives. This metric focuses on the accuracy of positive predictions. XGBoost demonstrated slightly higher precision compared to SVM and Random Forest with a score of 61.7%. This implies that when XGBoost predicted an image to be of a certain class, it was correct about 61.7% of the time.

Recall (or Sensitivity) is the ratio of true positives to the sum of true positives and false negatives. This metric shows how well a model can identify positive cases. SVM outperformed the other models in this regard with a recall score of 67.0%. This means SVM was able to correctly identify 67% of the positive cases.

The F1-score is the harmonic mean of Precision and Recall and gives a balanced measure of the model's performance. An F1-score closer to 1 indicates better performance. In this instance, SVM has the highest F1-score of 61.6%, indicating that it has a good balance of precision and recall.

### MODEL OPTIMIZATION

#### A. Hyperparameter Optimization
Hyperparameter optimization was performed for the XGBoost model using GridSearchCV from sklearn.model_selection. The goal is to find the best combination of hyperparameters to improve XGBoost performance on the cats and dogs images.

A grid search was conducted with cross-validation to determine the optimal hyperparameters for the XGBoost model. The necessary libraries, XGBClassifier from sklearn.svm and GridSearchCV from sklearn.model_selection, were imported. By defining the hyperparameters (n_estimators, learning_rate, max_depth, and colsample_bytree) along with their possible values, a new XGBoost model was created. The best hyperparameters and their corresponding score were then obtained from the grid search results.

#### B. Results of Hyperparameter Optimization
The best hyperparameters selected for the XGBoost model: {'colsample_bytree': 0.3, 'learning_rate': 0.01, 'max_depth': 5, 'n_estimators': 300}
Best Score: 0.627

Hyperparameter optimization using GridSearchCV helped identify the best hyperparameters for the XGBoost model. The optimal hyperparameters 'colsample_bytree': 0.3, 'learning_rate': 0.01, 'max_depth': 5, 'n_estimators': 300 resulted in a score of 0.627.

### EXPERIMENTATION WITH CNN

#### Data Preparation
The images in the dataset have been preprocessed into a suitable format for use in a Convolutional Neural Network (CNN). As part of data preprocessing, the images were resized to a fixed size (350x350 pixels) and normalized to have pixel values in the range [0,1] (done by dividing by 255, the maximum pixel value).

#### Image Data Generator
An ImageDataGenerator was used to automatically load the images from the dataset, convert them into a format suitable for input into a CNN, and apply any necessary data augmentation. The ImageDataGenerator was used to create both a training generator and a validation generator, each loading images from the dataset.

#### Model Building and Training
The model was built using TensorFlow's Keras API and consists of multiple convolutional layers, max-pooling layers, and dense layers. The model was compiled using 'binary_crossentropy' as the loss function, 'rmsprop' as the optimizer, and accuracy as the evaluation metric.

Over the 15 epochs, the model demonstrated significant learning, with the training accuracy generally increasing over time. The model achieved a final training accuracy of 98%.

#### Model Evaluation
However, the validation accuracy varied over time, with a final value of approximately 48%. The loss for both the training and validation sets generally decreased over time, but the validation loss demonstrated higher variability.



## CONCLUSION

### A. Summary of the Approach and Findings
The aim was to construct an image classifier that could accurately differentiate between images of cats and dogs. Machine learning algorithms, including Support Vector Machine (SVM), Random Forest, and XGBoost, and a CNN was used. These algorithms were trained on features extracted from the images.

During the preprocessing stage, images were loaded, displayed, normalized, and resized. Principal Component Analysis (PCA) was conducted to reduce the dimensionality of the dataset, which can help to simplify the model and decrease computation time.

The CNN performed better than all the other models in terms of accuracy. It is evident that the model is performing better on the training data compared to the validation data. This indicates that the model might be overfitting to the training data and not generalizing well to unseen data.

Strategies to improve the model's performance might include applying regularization techniques or adjusting the model architecture or hyperparameters.

In conclusion, this experiment demonstrates the capabilities of CNNs and other machine learning algorithms for image classification tasks. With further optimization and fine-tuning, the performance of this model on the cats and dogs classification task could likely be improved.

### B. Challenges and Limitations
Several challenges and limitations were encountered during this project, including:
- Overfitting: The models, especially the Random Forest and XGBoost models, were prone to overfitting due to the high-dimensional nature of image data.
- Image Data Handling: Traditional machine learning algorithms like the ones used in this project aren't inherently designed to handle image data and may lose important spatial information during the feature extraction process.
- Computational resources: The CNN was run on a personal computer, and this made the PC heat up uncontrollably.
