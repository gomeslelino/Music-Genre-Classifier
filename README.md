# Music-Genre-Classifier
The GTZAN dataset consists of 1000 audio tracks each 30 seconds long. It contains 10 genres, with 100 tracks per genre. The objective of this project was to build a deep learning model that can classify music tracks into different genres using the GTZAN dataset and librosa.

With the famous  GTZAN  music genre dataset, a  Deep Learning  algorithm was build using  librosa,  tensorflow, and  scikit-learn  to identify features inside the music files that will be used for classification and build an application in Python that automatically classifies different musical genres from an audio snippet. 

The project also allowed me to classify which musical genres possess a lot of sonic similarities and which are further apart from a mathematical standpoint.

## Data Preprocessing

The GTZAN dataset was loaded and the features from audio files (Mel-frequency cepstral coefficients (MFCCs), chroma features and mel spectogram features) were extracted thorugh a function created with the librosa library. This function returns a list of extracted features. Then, the dataset was split into training, validation, and test sets.

> Training set: 639<br>
> Validation set: 160<br>
> Test set: 200<br>
> Number of classes: 10<br>

Then, a function to plot the raw waveform and the Short-Time Fourier Transform (STFT) spectrogram for the audio files was defined, iterating over each genre. For each genre, it selects the first audio file in that genre and plots the waveform and spectogram for it.
The goal is to initially visualiza patterns in the genres, the comparison will be later refined when the correlation and similarities between them is better defined. 

<p align="center">
  <img width="1022" height="650" src="https://github.com/gomeslelino/Music-Genre-Classifier/blob/main/Pictures/Spectogram.png">
</p>

The STFT Spectrogram is a visualization of how the frequency content of a signal changes over time, while the Raw Waveform represents the amplitude of the signal over time without frequency information.

## Model Building

The deep learning model was built using a convolutional neural network (CNN). The architecture consists of convolutional layers followed by max-pooling layers, which are common choices for processing sequential data like audio signals.

The total number of trainable parameters is within a reasonable range, suggesting that the model is neither too simple (which may lead to underfitting) nor overly complex (which may lead to overfitting). 

A dropout layer was included to mitigate overfitting by randomly dropping a fraction of input units during training. This regularization technique helps improve the generalization ability of the model.

The loss function, optimizer, and metrics chosen during model compilation (sparse categorical cross-entropy, Adam optimizer, and accuracy metric) are suitable for the classification task at hand. They are commonly used for similar tasks and have been shown to yield good results in practice.

## Model Training

This neural network model was trained for 50 epochs with a batch size of 32. During training, it also evaluates the model performance on a separate validation set (X_val and y_val). The training progress is monitored, and the loss and accuracy metrics are recorded for both the training and validation sets at each epoch.

The results were plotted in two graphs:

> Training and Validation Loss: This graph visualizes how the loss (error) changes over epochs for both the training and validation sets. It helps to assess whether the model is overfitting or underfitting by comparing the loss values between the training and validation sets. 

<p align="center">
  <img width="567" height="453" src="https://github.com/gomeslelino/Music-Genre-Classifier/blob/main/Pictures/Loss1.png">
</p>

The Validation Loss wasn't decreasing toward zero like Training Loss, ideally, we should have a decreasing training loss coupled with a decreasing validation loss, indicating that the model is learning well. Learning Rate and Batch Size were calibrate to improve results.

> Training and Validation Accuracy: This graph shows how the accuracy of the model changes over epochs for both the training and validation sets. It provides insights into the model's ability to generalize to unseen data.

<p align="center">
  <img width="567" height="453" src="https://github.com/gomeslelino/Music-Genre-Classifier/blob/main/Pictures/accuracy1.png">
</p>

Increasing accuracy on both sets indicates that the model is learning useful patterns from the training data and can generalize well to new data. Some iterations were run by calibrating the parameters and the best result achieved was stabilization at 0,7.

## Model Evaluation

Metrics such precision, recall, and F1-score were calculated, all of them between 0.6-0.7, these were the best results obtained after many different fine-tuning iterations:

> Precision: 0.68<br>
> Recall: 0.64<br>
> F1-score: 0.64<br>

When the model is trained without data augmentation, however, we obtain:

> Precision: 0.71<br>
> Recall: 0.70<br>
> F1-score: 0.69<br>

The confusion matrix was also generated and corroborate this notion that there are a lot of musical similarities between the genres, so it is difficult to develop a model with much higher accuracy and precision:

<p align="center">
  <img width="833" height="743" src="https://github.com/gomeslelino/Music-Genre-Classifier/blob/main/Pictures/Confusion_Matrix.png">
</p>

Finally, I short python function was written to identify which genre was the most confused, that being "country", and which genre was the least ("disco"), a breakdown of country's similarities can be seen below:

> Genres confused with 'country':<br>
> blues        7<br>
> rock         4<br>
> disco        2<br>
> pop          2<br>
> jazz         1<br>
> reggae       1<br>
> classical    0<br>
> hiphop       0<br>
> metal        0<br>

## Conclusion
This project aimed to demonstrate the application of deep learning in music genre classification. By following the outlined steps, one can build, train, and evaluate a deep learning model capable of accurately classifying music tracks into different genres. The best achieve precision was around 0.71. This is due to the fact that there are a lot of sonic similarities between the genres. Country is the genre with most similarities amongst other genres, which Disco is the genre with the least similarities.
