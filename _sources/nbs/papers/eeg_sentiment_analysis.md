# Sentiment Analysis from EEG Sginal Data

## Prediction of advertisement preference by fusing EEG response and sentiment analysis

::::{grid}
:gutter: 1

:::{grid-item-card} To Read
[Paper](https://www.sciencedirect.com/science/article/pii/S0893608017300345)
:::

The aim is to predict rating of video-advertisements based on a multimodal framework combining physiological analysis of the user and global sentiment-rating available on the internet. Textual contents from review comments were analyzed to obtain a score to understand sentiment nature of the video. A regression technique based on Random forest was used to predict the rating of an advertisement using EEG data. Finally, EEG based rating is combined with NLP-based sentiment score to improve the overall prediction.

Signal preprocessing: Independent Component Analysis (ICA) to remove artifacts, Moving Average (MA) Filter to smooth.

---

## Recognition of human emotions using EEG signals: A review

::::{grid}
:gutter: 1

:::{grid-item-card} To Read
[Paper](https://www.sciencedirect.com/science/article/abs/pii/S001048252100490X)
:::

Electroencephalography (EEG) signals have become a common focus of such development compared to other physiological signals because EEG employs simple and subject-acceptable methods for obtaining data that can be used for emotion analysis.

Physiological signals: electroencephalography (EEG), electrocardiography (ECG), galvanic skin response (GSR), blood volume pulse (BVP), respiration rate (RT), skin temperature (ST), electromyography (EMG) and eye gaze.

Some of the physiological signals, like GSR, EMG, RT, and ECG that have been used to detect emotions have generally proven accurate in detecting only certain specific emotions.

EEG is a non-invasive method employed to monitor brain states and responses and has been used to monitor and diagnose seizures. The alpha wave is notably present in the occipital portion of the cortex when a person is stress-free and calm. Beta waves occur in the central and frontal parts of the brain when a person is actively working or thinking. The gamma frequency band brainwave is generally observed in the context of anxiety, sensory processing, and emotional stress.

Standardized EEG nomenclature, each location is designated thus to identify the lobes: frontal (F), temporal (T), occipital (O), parietal (P); letters A or M rep- resents the ears. Electrode position is indicated by a number, with electrodes on the right side of the brain is designated by even-numbers (thus F4, F8, P4, T6, etc.), and left side odd-numbers (F7, F3, T5, P3, etc.). ST and GSR can detect arousal levels, while EMG
measures valence.

Public datasets: DEAP, DREAMER, SEMAINE, MELD, IAPS, ASCERTAIN, MANHOB-HCI, DECAF, SEED, AMIGOS, CAPS and IN- TERFACE.

Some researchers observed that audio-video clips are much more efficient to trigger emotions compared to other stimuli.

To monitor EEG signals, electrodes are placed over the scalp in a configuration that obtains electrical signals with the best possible spatial resolution and signal to noise ratio (SNR). The number of electrodes must be kept low to avoid system difficulty.

In recent studies, many studies have applied deep learning classification methods for estimating emotions. It is found that above 70% of the studies used SVM, which is relatively easy to use. NB, KNN, DNN, and ANN ob- tained better results, typically with more than 80% accuracy. Spiking neural network and hierarchical fusion CNN also provide better performance than the normal CNN.

Researchers have recommended nonlinear features because nonlinear features would be more effective in knowing human emotions.

---

## A Deep Evolutionary Approach to Bioinspired Classifier Optimisation for Brain-Machine Interaction

::::{grid}
:gutter: 1

:::{grid-item-card} To Read
[Paper](https://www.hindawi.com/journals/complexity/2019/4316548/)
:::

This study suggests a new approach to EEG data classification by exploring the idea of using evolutionary computation to both select useful discriminative EEG features and optimise the topology of Artificial Neural Networks. An evolutionary algorithm is applied to select the most informative features from an initial set of 2550 EEG statistical features. Optimisation of a Multilayer Perceptron (MLP) is performed with an evolutionary approach before classification to estimate the best hyperparameters of the network. Deep learning and tuning with Long Short-Term Memory (LSTM) are also explored, and Adaptive Boosting of the two types of models is tested for each problem.

---

## A Study on Mental State Classification using EEG-based Brain-Machine Interface

::::{grid}
:gutter: 1

:::{grid-item-card} To Read
[Paper](https://ieeexplore.ieee.org/document/8710576)
:::

By using Muse headband with four EEG sensors (TP9, AF7, AF8, TP10), categorized three possibles states: relaxing, neutral, and concentrating. Created five individuals and sessions lasting one minute for each class.

Extracted five signals from EEG headband: alpha, beta, theta, delta, gamma. 10 fold cross validation. Results show that only 44 features from 2100 are necessary when used classifiers such as Bayesian NNs, SVM, and RF.

A major challenge of brain machine interface applications is inferring how momentary brain states are mapped into a particular pattern of brain activity. Lack of data issue: signals are complex, non-linear, non-stationary, and random in nature. In short time intervals signals are considered stationary, so, the best practice is to apply short-time windowing technique to detect local discriminative features. 

A Spking Neural Network was developed to classify seizuredetection based on statistics extracted from EEG streams with high accuracy of 92.5%. RF 82%, Bayesian classfier 92-97%, NN 64%.

Feature selection algorithms are applied w/ different combinations of ML algorithms: OneR, Information Gain, Correlation, Symmetrical Uncertainty, Evolutionary Algortihm. 

---

