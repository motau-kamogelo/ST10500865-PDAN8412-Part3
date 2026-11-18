# ST10500865-PDAN8412-Part3

 Table of Contents
Executive summary	1
1. Introduction and project context	1
2. Dataset selection and suitability	1
3. Analysis plan and methodology	2
4. Tools and implementation environment	3
5. Exploratory data analysis and data preparation	4
6. Feature selection and representation	6
7. Model design and training	6
7.1 Baseline CNN	6
7.2 Improved CNN	7
8. Model evaluation and results	8
9. Interpretation and discussion	12
10. Limitations, ethics and risk	13
11. Recommendations and conclusion	13
12. Reproducibility and submission artefacts	14
References	14

List of Figures
Figure 1: CIFAR-10 training images per class	6
Figure 2: One training example from each class	6
Figure 3: Pixel intensity distribution	7
Figure 4: Baseline CNN training history	9
Figure 5: Improved CNN training history	9
Figure 6: Baseline CNN - Confusion Matrix	11
Figure 7: Improved CNN - Confusion Matrix	12
Figure 8: Model evaluation and results	13
List of Tables
Table 1: Analysis plan and methodology	4
Table 2: Tools and implementation environment	5
Table 3: Exploratory data analysis and data preparation	7
Table 4: Model evaluation and results	10
Table 5: Model accuracy scores	10


 Executive summary
This report describes the creation and analysis of 2 CNN models for multi-class image classification. This study was conducted on the CIFAR-10 dataset, which consists of 60,000 labeled color images from 10 object classes. The basic CNN model showed a test score of 72.98% (test loss of 0.8134). The new CNN model with modifications such as data augmentation, batch normalisation, dropout, and control of learning rate was able to show a test accuracy of 80.79% (test loss of 0.5700). This means that the new model improved the results by 7.81% and reduced the loss level by 0.2434.
The new model is a better model for this dataset and this task in particular since its macro precision equaled 0.8144, while the macro recall and the macro F1-score were 0.8079 and 0.8061, respectively. Thus, it is important to note that this model demonstrates a prototype of educational image classification and not a facial recognition system. The authors of the dataset do not represent persons, their faces or personalities in the images from CIFAR-10, which denotes that it is impossible to employ these results in the context of the intended security implementation.
1. Introduction and project context
The publisher is looking at using image-recognition technology, with facial recognition being a possible application in the future. The proposal of the project allows using a first model for image recognition that is not meant for face detection. Therefore, this project describes the tasks as classification of images in ten specific categories.
Using a CNN for image classification is a valid choice since it allows learning about the local spatial characteristics of images. The results of the work are evaluated in terms of overall accuracy as well as class-specific metrics such as recall, precision, the F1-score, loss curves, and the Confusion matrix. The report is divided into the following sections: dataset justification, plan for analysis, exploratory data analysis (EDA), model training, evaluation, and suggestions.
2. Dataset selection and suitability
CIFAR-10 is the dataset that we will choose for our purposes, which is provided through the TensorFlow/Keras dataset. It comprises 60,000 RGB images of size 32 x 32 pixels, split into 50,000 training images and 10,000 test images. We have ten labels in CIFAR-10, namely, airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck, this means each class has in total 6,000 images in CIFAR-10 and in the training part, we will have 5,000 in the training part and 1,000 in the testing one.
CIFAR-10 will serve our purposes perfectly since it has more than 10,000 records as stated in the brief, it is open to the public, it has labeled data, it has fixed and convenient image shape and it allows to implement supervised learning methods in a reproducible way. The equal distribution of classes allows to evaluate the quality of models less complicated and makes it possible to avoid the situation when the accuracy of the model is affected by the dominant class.
According to the data quality checks in the notebook, the image is represented as unsigned 8-bit where its pixel values vary from 0 to 255 and ten labels are as expected. We need to note, however, that the dataset is modest in terms of image resolution, which means it may contain ambiguous examples and does not account for all variations in images, quality, illumination or camera positions.
Thus, this dataset can be applied to learning as well as assessing general object classification but cannot be used for training or validation of facial recognition systems, identity matching systems, access systems or for monitoring through CCTV. This point is essential for using the results correctly.
3. Analysis plan and methodology
Before training the models, analysis carried out in order to ensure a comprehensive understanding of the data set, as well as the uphold of the division of the data into three parts (training/validation/testing) and evaluation being conducted on both overall and class-specific levels.

Stage	Planned activity	Purpose / output
1. Data acquisition	Load CIFAR-10 using TensorFlow/Keras; record source, size, labels, image shape and data types.	Establish provenance, scope and suitability.
2. Data validation	Check dimensions, pixel range, label range, missing values and class counts.	Identify malformed records and quality concerns.
3. EDA	Plot class distribution, sample images and pixel-intensity distribution; inspect image variety.	Identify balance, visual patterns and limitations that affect modelling.
4. Feature learning	Use CNN convolutional filters to learn spatial features directly from pixels; avoid manual p-value selection.	Capture edges, textures and higher-level visual patterns.
5. Data preparation	Convert pixels to float32 and scale to [0, 1]; create a stratified 40,000/10,000 train-validation split from the 50,000 training images.	Stabilise optimisation and preserve class proportions.
6. Baseline model	Train a sequential CNN with convolution, pooling, dense and dropout layers; monitor validation loss and use early stopping.	Create a measurable benchmark.
7. Model improvement	Add augmentation, batch normalisation, dropout and adaptive learning-rate control; retain early stopping.	Address overfitting and improve generalisation.
8. Evaluation	Evaluate once on the held-out 10,000-image test set; calculate accuracy, loss, per-class precision/recall/F1 and confusion matrix.	Compare models and identify class-specific errors.
9. Reporting	Present plots, metric tables, limitations, reproducibility details and recommendations.	Communicate evidence and avoid claims beyond the experiment.
Table 1: Analysis plan and methodology
The notebook utilized a constant random seed in order to ensure that results could be reproduced. The test set was considered separately from the training and validation sets and utilized only for final model evaluation. Spark has been used for distributed-style data inspection and training labels aggregation, according to requirements; TensorFlow/Keras has been used for training the CNNs, since model training is not required to be performed in Spark according to the requirements..
4. Tools and implementation environment
Component	Implementation recorded in notebook
Programming environment	Python 3.11 (Windows)
Deep-learning framework	TensorFlow 2.21.0 / Keras
Distributed data processing	Apache Spark / PySpark 4.2.0
Data and numerical processing	NumPy 2.4.6 and pandas
Visualisation	Matplotlib and Seaborn
Model evaluation	scikit-learn classification metrics and confusion matrix
Execution device	Native Windows CPU; TensorFlow reported no native-Windows GPU support.
Table 2: Tools and implementation environment
5. Exploratory data analysis and data preparation
Prior to fitting the model, the dataset underwent inspection. The training set was made up of 50,000 images, while the test set was made up of 10,000 images. Each image had three color channels and dimensions of 32 × 32 pixels. The image tensors were composed of uint8 values ranging from 0–255. All ten anticipated classes were accounted for.
The class-frequency analysis showed that exactly 5,000 training samples were made available for each of the ten classes. This balanced distribution is essential because it means that the basic accuracy is not raised artificially by one class having a larger representation than the others. The Spark DataFrame aggregation confirmed having 5,000 training records for each of the labels and 50,000 records in total.
The notebook verified and discovered that there were no images missing and no labels missing. Since the image data is already a complete fixed-size tensor with no missing data, the data did not have to be replaced. Therefore, none of the records were discarded. The pixel values were converted into float32 and divided by 255 creating a normalized range from 0.0 to 1.0.
The original 50,000 images were split into 40,000 training images and 10,000 validation images using stratification, whereas the 10,000 test images remained separate and untouched throughout the model fitting and validation process..
 
Figure 1: CIFAR-10 training images per class
 
Figure 2: One training example from each class
 
Figure 3: Pixel intensity distribution

EDA check	Observed result	Interpretation
Training records	50,000	Exceeds the 10,000-record minimum.
Test records	10,000	Independent final evaluation set.
Image dimensions	32 × 32 × 3	Small RGB images; limited fine detail.
Pixel data type/range	uint8; 0–255	Valid image intensity range.
Training class counts	5,000 per class	Balanced training classes.
Missing image values	0	No imputation required.
Missing labels	0	No missing target labels.
Normalised range	0.0–1.0	Suitable numeric scale for neural-network optimisation.
Table 3: Exploratory data analysis and data preparation
The EDA results have guided the selection of models. The balance of classes makes it possible to apply the accuracy metrics along with the macro-average metrics. The resolution of images led to the selection of the convolutional model that is capable of learning the hierarchical spatial features. Normalisation was used in advance of training, and augmentation is employed in the new model to let the model confirm its learning via the real-world transformation of the images.
6. Feature selection and representation
The conventional method of selecting features, including the use of p-value or stepwise inclusion/exclusion methods, is inherently unsuitable when raw image tensors are processed in the convolutional neural network (CNN). On the contrary, the algorithm extracts useful information from the given pixels. The first layers of convolution are focused on learning uncomplicated changes in pixel values, whereas the last layers aggregate learned theory into complex forms. 
In this CNN image-processing, every RGB component of an image was utilized along with the full spatial layout of the image size of 32 × 32 pixels. No specific actions were taken to get rid of any features manually. In the course of the training, the convolutional filters, pooling layers, and trained weights took care of the extraction of features, which seems reasonable since the relative position of the pixels is of significance when it comes to images.
7. Model design and training
7.1 Baseline CNN
The baseline model adopted a sequential approach with 3 convolutional layers (32, 64 and 128 filters), 3 × 3 kernels, ReLU activations and max-pooling. After the feature maps extraction, the model was flattened into a 128 units dense layer that was followed by a dropout layer and a 10 outputs layer. The total number of parameters in the baseline was 356,810.
In the course of training, the baseline was observed through the validation set so that the early stopping could monitor the validation loss with a patience of four epochs and save the optimal weights.
 
Figure 4: Baseline CNN training history
7.2 Improved CNN
The enhanced architecture featured the usage of data augmentation, which involves the applications of random horizontal flipping and translation, batch normalisation, dropout, convolutional layers, and global pooling. The total number of parameters was 157,514 (157,322 of them are trainable and 192 non-trainable). The goal of the architecture was to improve the process of generalisation.
The enhanced architecture included early stopping allowing for a patience of five epochs and the callback ReduceLROnPlateau that changes the learning rate when validation loss does not improve any longer. The goal of these changes is to lessen the model’s sensitivity to object placement and increase stability of the optimisation process.
 
Figure 5: Improved CNN training history
Both models were trained in Python with TensorFlow/Keras. The notebook saves the trained models as .keras files and exports a model-comparison CSV, supporting reproducibility and further inspection.
8. Model evaluation and results
The test dataset was composed of a total of 10,000 images, ensuring that there were 1,000 samples belonging to each of the classes. Accuracy was used as the primary measure of correct classification. Loss offers an alternative measure of prediction error. Precision displays the amount of class instances that were correctly predicted, recall shows the degree of actual instances properly classified, and F1 score is the mean of both precision and recall scores. Macro average allows for each of the classes being represented equally due to the nature of this data, which is well-balanced.
.
Metric	Baseline CNN	Improved CNN	Change
Test accuracy	72.98%	80.79%	+7.81 percentage points
Test loss	0.8134	0.5700	−0.2434
Macro precision	0.7282	0.8144	+0.0862
Macro recall	0.7298	0.8079	+0.0781
Macro F1-score	0.7280	0.8061	+0.0781
Table 4: Model evaluation and results
The advanced CNN was more effective than the baseline in all of the aggregate assessment metrics taken into account. The test accuracy was increased from 72.98% to 80.79% and the test loss decreased from 0.8134 to 0.5700. The increase in the macro F1-score was from 0.7280 to 0.8061. The absolute improvement in accuracy was 7.81 percentage points, which should be considered a substantial applied improvement for this study, although statistical significance tests and seed repetitions were not conducted, hence the change cannot be called statistically significant.
 
Table 5: Model accuracy scores
In the enhanced version, the automobile has a notable F1-score of 0.9121 and recall of 0.9440. Ship achieved the highest precision 0.9393, whereas cat has the lowest recall of 0.6010. The overall success of the category is represented by the F1-score of 0.6572 indicating that cat images are hard to classify. Frog has both high recall of 0.9370 and low precision of 0.6467 indicating that there are many frogs predicted but which actually belong to other classes.
 
Figure 6: Baseline CNN - Confusion Matrix
 
Figure 7: Improved CNN - Confusion Matrix
Off-diagonal cells should be used to interpret the confusion matrix. After analyzing the confusion matrix generated using the improved model, list out the major class-confusion pairs. The possible visual reasons should be presented only if they are justified by the figure and the confusion matrix rather than guessing actual number of confusions from the results of classification.
 
Figure 8: Model evaluation and results
9. Interpretation and discussion
The baseline model demonstrated that the gap between training performance and validation performance continued to increase as the training proceeded. For instance, after epoch 8, the accuracy of the training was equal to 78.08%, while the accuracy of validation on the same epoch was equal to 73.12%; furthermore, the accuracy of validation remained steady afterwards, which corresponds to the situation with overfitting. The accuracy on test equaled 72.98%, which is similar to the validation results.
The advanced model attained the accuracy on the test level of 80.79% and the same percentage for the validation. The accuracy during the final epoch of training was smaller than the accuracy of validation, and this is acceptable because augmentation and dropout create more difficulty to the training scheme.
The outcomes received from the new method confirm the utilization of the designed architecture together with the trained model. Still, different changes took place at the same time, such as the use of augmentation, utilizing batch normalization, dropout, global average pooling and correcting a learning rate. Thus, the influence of each of the methods cannot be considered on its own. One can apply the ablation study in which one of the methods is changed at a time.
The metrics on the class level show that overall accuracy shows some differences. The model performs quite well in the recognition of cars and ships while it has significant difficulties in recognizing cat images and giving predictions about the frogs together with other classes.
10. Limitations, ethics and risk
The chief limitation lies in the disparity between the published dataset and the publisher’s long-term aspirations for facial recognition. CIFAR-10 is a focused dataset for object classification comprising 32 × 32 images. It lacks anybody, face or identity, access control events, or video footages from the CCTV. Moreover, it does not demonstrate the conditions of real-world lighting, camera positions, motion blurriness, or obstructions along with the demographics variations in the quality of surveillance images.
The model predicts ten categories of objects but does not detect the person. Hence, its accuracy which is 80.79% should not be taken as the measure of the efficiency of facial recognition accuracy or its safety. Furthermore, the experiment didn’t provide estimation of operational latency, robustness against changes in the distribution and possible adversarial attacks, personal privacy, demographic fairness and impact.
If a facial-recognition study is to be done in the future, it must be performed within lawful guidelines, under valid authority and with data that will be representative of the target population and environment. Prior to obtaining or processing any facial image, the publisher must conduct legal and privacy review, check POPIA obligations and status of biometric information, clearly indicate the purpose and legal ground, limit data collection and retention, secure the data, evaluate the occurrence of differential error rates and provide appropriate human oversight. Only automated output must not be used for making consequential access decisions.
11. Recommendations and conclusion
In terms of educational image classification purpose, improved CNN would be preferred, as it has reached a higher test accuracy and test loss rate and showed better macro precision, recall, and F1-score. The findings show that CNNs can build a valid image representation and distinguish various types of objects based on labelled pictures.
The model needs to keep its status of an educational model. It shouldn't be used for facial recognition, identity confirmation, and access control. The author of the paper should define the applicability of the system and, thus, them conduct an independent feasibility, privacy, security, ethics, and real-life testing study using the appropriate data and tool for evaluation.
The improved CNN has achieved the test accuracy of 80.79%, while baseline one was 72.98%. The 7.81 percentage points growth means improvement in performance on CIFAR-10 data.
12. Reproducibility and submission artefacts
The principal documentary proof is represented by the Jupyter Notebook, which comprises the Python code, Markdown notes, the aggregated data processed through Spark, an exploratory data analysis, model definitions, model training process and evaluation results. The outputs directory has the baseline and updated .keras model file as well as a model comparison CSV. All report tables and figures are to be produced from the final execution of the notebook prior to PD submission.
As a guarantee of reproducibility, it is necessary to store the notebook, exported models, model comparison CSV, link to the dataset and this report in the project repository. The date of the last execution and library versions shall also be included in the notebook.

 References
[1] A. Krizhevsky, “Learning multiple layers of features from tiny images,” University of Toronto, 2009. [Online]. Available: https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf. [Accessed: 17-Nov-2026].
[2] TensorFlow, “CIFAR-10 small images classification dataset,” TensorFlow API documentation. [Online]. Available: https://www.tensorflow.org/api_docs/python/tf/keras/datasets/cifar10. [Accessed: 17-Nov-2026].
[3] TensorFlow, “Conv2D layer,” TensorFlow API documentation. [Online]. Available: https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv2D. [Accessed: 17-Nov-2026].
[4] TensorFlow, “Data augmentation,” TensorFlow tutorials. [Online]. Available: https://www.tensorflow.org/tutorials/images/data_augmentation. [Accessed: 17-Nov-2026].
[5] Apache Spark, “Spark SQL and DataFrames,” Apache Spark documentation. [Online]. Available: https://spark.apache.org/docs/latest/sql-programming-guide.html. [Accessed: 17-Nov-2026].
