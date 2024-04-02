# Emotion_Classification_ML
 Developed ML models (Logistic Regression, SVM) to classify text-based emotions, achieving 80%+ accuracy. Enhanced sentiment analysis with advanced text preprocessing and feature engineering, identifying key emotional expression patterns. 

Emotion Classification using Machine Learning
Team Members :
Aarushi Slathia Laya Harwin Siva Krishna Gundam
aslathia@iu.edu lharwin@iu.edu sgundam@iu.edu

**ABSTRACT**

Emotion is essential to human communication, thus emotion classification has a wide variety of applications in the domains of business, healthcare, and education.The goal of emotion classification is to categorize human feelings into various parts of the emotion spectrum, such as joy, fear, love, sadness, and anger. After analysis of these classes, we saw the potential in applying a Machine Learning(ML) based approach to solve this task. Our strategy was to thoroughly investigate various Machine Learning implementations of emotion categorization in order to better comprehend its practical applications. On a dataset of 20k samples divided into six different emotion classes, we trained various machine-learning models. We then evaluated the performance of the models by tuning the hyperparameters and observed some interesting results. The logistic regression model achieved an optimal performance with an accuracy of 87%.

**KEYWORDS**

Emotion classification, Data mining, Machine learning, Multi-class classification, Word cloud Logistic Regression, Stopwords, N-grams visualization

**INTRODUCTION**

Emotions are a vital part of communication which are conveyed through words, gestures, and facial expressions. Emotion classification, also known as emotion categorization, is the process of identifying emotions and assigning them to the appropriate category. Emotion classification can also be considered as an extension of Sentiment Analysis, where instead of classifying sentiment into polarities of positive, negative, and neutral, we classify the text into different categories of emotions. We have considered five categories of emotions in this project: sadness, anger, fear, joy, and love. In this work, we focused on analyzing the performance and evaluating the results of various Machine Learning algorithms: Logistic Regression, Decision Tree, K Nearest Neighbor, Multi-layer perceptron, Linear and Non-linear SVM on a common textual Emotion classification dataset collected from Kaggle. From our observations, we found that the Logistic Regression model outperformed all the other Machine Learning algorithms with an accuracy of 87% and KNN performed the least with an accuracy of 67%. This report includes the following sections: Introduction, Background, Motivation and Challenges, Related work, Methodology(Dataset, Data Pre-processing, Exploratory Data Analysis, Modelling), Experiments and Results, Conclusion and Discussions.

**BACKGROUND, MOTIVATION, AND CHALLENGES**

In order to effectively communicate emotions, one must use words, gestures, and facial expressions. Multiple parts of daily life, including text, multimedia, tweets, and other social media platforms, exhibit this characteristic. Every word or sentence a person writes will have their feelings attached to it. It is crucial to examine these feelings since doing so can help you better grasp the nature of the text. The classification of emotions has several uses, such as the early detection of sadness, the creation of more sympathetic chatbots, and improved user review analysis. Emotion Classification divides the text into many categories of emotions rather than dividing it into polarities of positive, negative, and neutral. Sadness, anger, joy, fear, and love are the various emotion categories that we have taken into account in this project. Large corpora of text data can now be trained on improved hardware resources for increased efficiency and quicker computational processing. This has enabled significant advancements in the field of emotion categorization, along with the use of extremely accurate pre-trained models. Still, there are a lot of challenges involved in the emotion classification task. The inability to detect double meaning, jokes, and innuendos and; the inability to account for regional variations of language and non-native speech structures are a few of the challenges involved with emotion classification.

**RELATED WORK**

The problem of classifying emotions in text has been investigated using a variety of methods, particularly in the fields of machine learning (ML) and deep learning. Previous research concentrated on applying traditional machine learning algorithms, such as the Hidden Markov Model (HMM) model of manually created temporal characteristics from raw data [1]. Gaussian Mixture Models (GMMs) were a high-level statistical characteristic used in other methodologies [2]. Convolutional Neural Networks (CNNs)-based deep learning approaches are more popular, and this has numerous advantages. Through the application of multi-modal techniques and Multi-Scale Convolutional Neural Networks that take into account both text and audio features to produce the output, researchers have demonstrated the effectiveness of distinguishing emotion from speech [3, 4]. It is feasible to further enhance the performance of the suggested model by adding an attention module constructed on top of the statistical pooling unit (a pooling unit used to further extract features from each modality). Similarly to this, Seyeditabari’s research demonstrates that traditional ML models disregard the text’s sequential nature and its context [5]. When attempting to generalize the model for emotion detection, this presents problems. They suggest a classifier built on a recurrent neural network (RNN) that can capture the context and sequential character of the text and, as a result, considerably enhance performance. The model is able to perform better with an average gain in the F-Score of 26.8 points on the test data by building a more informative latent representation of the target. Many computational bottlenecks are also experienced by RNN as highlighted in Minaee’s survey research on DL-based text categorization [6].

**METHODOLOGY**

In this research work, we have done a comparative study to show the performance of various Machine Learning models for emotion classification tasks. Studies were conducted using
Logistic Regression, Decision Tree Classifier, K nearest neighbor classifier, Multi layer perceptron Algorithm, Linear SVM and Non Linear SVM.

**Dataset:**

The dataset, which includes 20k samples with the features "sentence" and "label," was gathered from Kaggle. The dataset preparation method was influenced by CARER [7]. The
textual input data to be processed is contained in the "sentence" attribute, and the "label" attribute contains the various emotion labels corresponding to each phrase as shown in Table 1. Sadness, Anger, Joy, Fear, Love, and Surprise are among the six different emotion classes
represented in this dataset.

<img width="779" alt="Screenshot 2024-04-02 at 17 31 54" src="https://github.com/Aarushi253/Emotion_Classification_ML/assets/137174486/1581b96e-1362-4aed-924f-4a7d071caf4f">


**Data Pre-processing:**

The vital first step is to analyze and prepare the dataset before sending it to the predictive models. The dataset was in the form of three documents or text files. All the 3 text files(train, text and val) were initially concatenated into a single dataframe for easier pre-processing and cleaning. We then checked for missing values in the dataset. We have also checked the label/class/emotion category distribution in the dataset. There were very few samples of ‘Surprise’ class in the dataset compared to other classes. Hence, we have removed those samples from the dataset. Since, the labels were text and not numbers, we have encoded these text into numbers using label encoding. Coming back to the features, we first removed irrelevant characters, then tokenized the string and converted all the characters to lowercase. We also performed stemming to bring down all the words to a common base form. Once all these operations were applied, we used the vectorizer function from the scikit learn library to convert a collection of text documents to a matrix of token counts(Document term matrix). This is a required step before passing input to the model. Finally, we passed the processed data through a train test split function to get training and testing data. 80% of the data was used for training while the remaining was used for testing.

**Exploratory Data Analysis:**

Our dataset consisted of rows of sentences with labels as one of the emotion categories. To visualize the most frequent words in the sentences, we have plotted word clouds and also
added some barplots. Word cloud is plotted for each of the emotion classes as shown in Figure 1. Word clouds operate in a straightforward manner: they show specific words in a
bigger, bolder font the more frequently they appear in a source of textual data. Interestingly, we saw few of the most common words in all the classes were the same in the word cloud: really, feel, feeling etc. We have also added a data visualization technique i.e word count. While visualizing the most frequent words in the sentence, we observed that most of the sentences have stopwords in it. Stop words are words which occur frequently in a language. For example, ‘and’, ‘or’, ‘was’, etc. Removing these words will not change the semantics of a sentence and help us concentrate more on words that are relevant to the label. Here, we have used the ‘stopwords’ method provided by the ‘nltk’ library to filter out the most common stop words in English. Apart from this we also run the code iteratively and identify other stop words relevant to our data and exclude them too for visualization. During data analysis, we also came across one of the emotion classes(surprise), which had very few rows of sentences comparative to other emotion classes. Therefore, we have dropped this class during our data cleaning process. We have also plotted some barplots to visualize ngrams. NGrams is a feature used often in language processing to understand how words when occurring in ordered groups affect the labeling. It is a continuous sequence of n items in a given text or sentence. Here, N is a variable which tells the number of words in the sequence. N=2 means bigrams, N=3 means trigrams and so on. Here, we have plotted barplots for bigrams for all the classes.

<img width="757" alt="Screenshot 2024-04-02 at 17 53 43" src="https://github.com/Aarushi253/Emotion_Classification_ML/assets/137174486/3eafdf09-ac25-45ce-996c-e4628d7bc5fa">


**Modeling:**

We have implemented a total of 6 algorithms: Logistic Regression, Decision Tree, K Nearest Neighbor, Multilayer Perceptron, Linear SVM and Non-Linear SVM. Logistic Regression mostly addresses classification problems. It helps classifying data into distinct classes by analyzing the relationship from a given collection of labeled data. We can
consider this model as the most optimal solution for our classification as it gives the highest classification accuracy of 87% on the test set.
The K Nearest Neighbor algorithm had the least accuracy for our classification as the clusters in our data for classification are of varying sizes. KNN is time consuming and performs poorly with large complex datasets. It is a notoriously sluggish algorithm since it only takes into account its neighbors when classifying. We got an accuracy of 67% on the test data using KNN algorithm. Next, we considered the Decision Tree algorithm in our experiment. A decision tree is used to solve a problem where each leaf node corresponds to a class label and the internal node of the tree represents attributes. We got an accuracy of 81% with the decision tree algorithm on the test data.
We also experimented with Linear and Non-linear SVM algorithms. In an N-dimensional space (N being the number of features), the support vector machine algorithm seeks to locate a
hyperplane that clearly categorizes the data points. The accuracy for the Linear and Non-linear SVM algorithm were 85% and 82% respectively. Finally, we wanted to work with more complex algorithms like Multi-layer Perceptron(MLP) and utilize the capability of Neural Networks. A neural network approach called the multilayer perceptron discovers the connections between linear and nonlinear data. We ran the MLP Classifier for 300 iterations with Relu Activation function and Adam optimizer. The MLP algorithm was able to achieve a decent accuracy of 84% on the test set.

**EXPERIMENTS AND RESULTS**

To assess the effectiveness of our models, we have employed a variety of measures. The Kappa coefficient, Jaccard accuracy, F-Score, Precision, Recall, and Pearson Correlation are few of the common metrics used for multiclass classification tasks. These metrics are best suited for tasks of multi-class classification problems where we categorize the outputs into one of the emotion classes. Based on our requirements, we have used F- Score, Precision, Recall, and Accuracy metrics to compare the model performances. The formula for calculating these metrics are as shown below.

<img width="392" alt="Screenshot 2024-04-02 at 17 54 30" src="https://github.com/Aarushi253/Emotion_Classification_ML/assets/137174486/e511f99d-f31b-480b-8bf1-fa629f8c8c26">

Here, TP is True Positive, FP is False Positive, TN is True Negative and FN is False Negative. In our experiments, we have made use of the scikit-learn library for the implementation of machine learning models. We split the dataset as 80% for training, and 20% for testing. In the Logistic Regression model, we have set the multiclass parameter to multinomial and trained the model for 500 iterations. We were able to achieve an accuracy of 87% with a logistic regression algorithm. We also implemented Decision Tree, K-Nearest Neighbor(with 5 neighbors), Multi-layer Perceptron algorithm(with Relu Activation and Adam Optimizer), Linear SVM, Non-Linear SVM and tested the performance on emotion classification tasks. The performance of the K Nearest Neighbor algorithm was not up to the mark as it was able to achieve an accuracy of only 67%. Another observation was made in terms of changes in loss with increasing iterations of the multi-layer perceptron(MLP) model being run. The MLP algorithm was able to converge after 7-8 iterations as shown in the figure below(Figure 2) and the loss was almost constant after that.

<img width="617" alt="Screenshot 2024-04-02 at 18 03 26" src="https://github.com/Aarushi253/Emotion_Classification_ML/assets/137174486/9ab463fa-abf0-4e05-afdd-56588a44d1e5">

Table 2 below tells us about the data distribution for each emotion category with their respective labels. 

<img width="667" alt="Screenshot 2024-04-02 at 18 03 54" src="https://github.com/Aarushi253/Emotion_Classification_ML/assets/137174486/cb678719-ebdf-4543-a5c9-574e30908022">


The detailed result of each model/algorithm is shown below in Table 3,4,5,6,7
and 8.

<img width="850" alt="Screenshot 2024-04-02 at 18 05 05" src="https://github.com/Aarushi253/Emotion_Classification_ML/assets/137174486/21e99081-d83d-4e84-819b-0a758d729d70">

<img width="823" alt="Screenshot 2024-04-02 at 18 05 13" src="https://github.com/Aarushi253/Emotion_Classification_ML/assets/137174486/130f4106-df7f-46e2-9eb2-9493885190bc">

<img width="865" alt="Screenshot 2024-04-02 at 18 05 19" src="https://github.com/Aarushi253/Emotion_Classification_ML/assets/137174486/f25e199d-ee94-405e-94e4-1ad6ded87155">

<img width="841" alt="Screenshot 2024-04-02 at 18 05 24" src="https://github.com/Aarushi253/Emotion_Classification_ML/assets/137174486/f257fafc-4c02-4943-aa05-430ae077a1a1">


One interesting commonality that we observed in all the predictive models was that the
precision, recall and f1-score values were highest for labels that previously had the most
number of samples in the dataset. Here, class joy represented as class 2 and sadness
represented as class 4 have the highest number of samples in the dataset as shown in Table 2
below. It can be observed from table 3-8 that the precision, recall and f1 score values are
highest for these 2 classes for all the algorithms. This tells about the importance of the dataset
for any machine learning task.

The performance of each model is compared using barplot as shown in Figure 3 below. 

<img width="507" alt="Screenshot 2024-04-02 at 18 06 27" src="https://github.com/Aarushi253/Emotion_Classification_ML/assets/137174486/174f3961-75ef-444f-8b84-89dfbd209c30">

We can clearly see that the Logistic Regression model outperformed all the other algorithms. KNN algorithm performed the least among all the other algorithms. All other algorithms were also able to achieve a decent accuracy on the test set.

**CONCLUSION AND DISCUSSIONS**

Our experimentation was conducted on various Machine Learning algorithms: Logistic
Regression, Decision Tree, KNN, Multilayer Perceptron, Linear and Non-Linear SVM. The
findings show that the logistic regression model outperformed all other Machine Learning
Algorithms with an accuracy of 87%. Also, the K Nearest Neighbor algorithm has least
performance among all the algorithms. However, we believe there is a scope of improvement in
performance if we take various effective measures into account like regularization technique,
better dataset, more complex algorithms like deep learning and transformer based models like
BERT. We also need to take into consideration the potential impact a balanced dataset with
distinct samples would have on the training of the predictive models. This could lead to more
accurate models, thereby alleviating the low precision and recall scores of some of the emotion
classes. The findings discovered in our investigations are indicative of the potential future
research that may be conducted in the domain of emotion classification. The inclusion of voice
and facial data allows researchers to comprehensively predict emotion classes as well as
detect early signs or symptoms of mental health issues like depression. Making chatbots more
humane or empathetic is another application of emotion detection with the goal of making
chatbots capable of responding appropriately to the emotional needs of users. Companies can
create better products and services by employing emotion detection to gain a deeper
understanding of user reviews.

**REFERENCES**

[1] Tin Lay Nwe, Say Wei Foo, and Liyanage C De Silva. 2003. Speech emotion recognition
using hidden Markov models. Speech communication 41, 4 (2003), 603–623.

[2] Daniel Neiberg, Kjell Elenius, and Kornel Laskowski. 2006. Emotion recognition in
spontaneous speech using GMMs. In Ninth international conference on spoken language
processing.

[3] Yoon Kim. 2014. Convolutional Neural Networks for Sentence Classification.
In Proceedings of the 2014 Conference on Empirical Methods in Natural Language
Processing (EMNLP). Association for Computational Linguistics, Doha, Qatar, 1746–1751.
https://doi.org/10.3115/v1/D14-1181

[4] Zixuan Peng, Yu Lu, Shengfeng Pan, and Yunfeng Liu. 2021. Efficient Speech Emotion
Recognition Using Multi-Scale CNN and Attention. In ICASSP 2021-2021 IEEE
International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE,
3020–3024.

[5] Manish Munikar, Sushil Shakya, and Aakash Shrestha. 2019. Fine-grained sen- timent
classification using bert. In 2019 Artificial Intelligence for Transforming Business and
Society (AITB), Vol. 1. IEEE, 1–5.

[6] Shervin Minaee, Nal Kalchbrenner, Erik Cambria, Narjes Nikzad, Meysam Chenaghlu, and
Jianfeng Gao. 2021. Deep learning–based text classification: a comprehensive review.
ACM Computing Surveys (CSUR) 54, 3 (2021), 1–40.

[7] Elvis Saravia, Hsien-Chi Toby Liu, Yen-Hao Huang, Junlin Wu, and Yi-Shin Chen. 2018.
CARER: Contextualized Affect Representations for Emotion Recognition. In Proceedings
of the 2018 Conference on Empirical Methods in Natural Language Processing.
Association for Computational Linguistics, Brussels, Belgium, 3687– 3697.
https://doi.org/10.18653/v1/D18-1404
