<h1 align="center">Coronary Artery Disease Prediction using Ensemble of Electrocardiogram and Phonocardiogram with other Vitals</h1> <br>


## Table of Contents 

- [Introduction](#introduction)
- [Features](#features)
- [Results](#results)
- [Contributors](#contributors)
- [Build Process](#build-process)

## Introduction

Today, cardiovascular disease (CVD) kills the majority of people worldwide. Given the severity of this condition, the analysis and diagnosis of CVD becomes critical. In the presence of subject matter experts (SMEs), this task becomes trivial. On the other hand, a phonocardiogram (PCG) is almost always impossible without the aid of an SME to diagnose cardiovascular disease with the aid of an electrocardiogram (ECG). In both of the above cases, the deep learning method proposed in this paper can serve as a convenient tool to help the field of diagnosis in this regard. Electrocardiographic analysis, PCG is the main basis for exploring the heart health of patients. Various methods have been proposed to analyze these vital signs individually, however, this paper provides a comprehensive and detailed overview of the analysis of the above-mentioned collection of data. Minimizing false negative predictions is necessary for the medical field, otherwise these false predictions pose an immediate threat to patients.


## Features

1. CSV Data Prediction
2. ECG Data Prediction
3. Audio Data Prediction

## Introduction

UNDOUBTEDLY the heart is the core organ of the human body whose uninterrupted functioning is essential for the daily activities of humans. However, Coronary artery disease (CAD) a type of cardiovascular diseases (CVDs), substantially obstruct the heart functions and thus became a prime reason for deaths all around the world. According to many surveys conducted by the World Health Organization (WHO), 33% of all deaths are due to CAD [1]. It originates through the gathering of plaques beside the inner walls of coronary arteries that decreases the flow of blood towards the myocardium [2, 3]. In severe circumstances, the cracked plaques can entirely seal
the arterial lumen, which yields triggering of an acute Corresponding myocardial infarction. At present, coronary angiography, the gold standard in the clinical diagnosis of CAD, is an invasive technique that requires professional surgical procedures, considerable time, and cost. Electrocardiogram (ECG), Phonocardiogram (PCG), and other vitals like resting heart rate, body mass index (BMI) function as important assets in monitoring the health of the heart. The electrocardiogram includes the "P" wave, "QRS" wave group and "T" wave, showing heart activity in various parts of the body. The first electrical signal on a normal ECG comes from the atrium, called the "P" wave. Although there is usually only one P wave in most ECG leads, the P wave is the sum of the electrical signals from the two atria, usually superimposed. Then there is a short physiological delay because the atrioventricular (AV) node slows the electrical depolarization. before it enters the ventricle. This delay is the cause of the PR interval, which is an abbreviated period when no electrical activity is seen on the ECG, represented by a horizontal straight line or "equipotential" line. The depolarization of the ventricle usually results in the largest part of the ECG signal (because the muscle mass in the ventricle is larger), which is called the QRS complex. Q wave is the first initial downward or "negative" deflection. Then the R wave is the next upward deflection (assuming it crosses the equipotential line and becomes "positive"). The S wave is then deflected downward the next time if it crosses the equipotential line and becomes temporarily negative before returning to the equipotential baseline [2]. In the case of the ventricle, there are also electrical signals that reflect the repolarization of the myocardium. This is shown as the ST segment and the T wave. The ST segment is normally isoelectric, and the T wave in most leads is an upright deflection of variable amplitude and duration.

Along with the ECG, Phonocardiogram (PCG) can be used to monitor the heart. PCG is a high-fidelity recording plot of the phonetics made by the heart with the help of the machine called the phonocardiography; thus, phonocardiography is used to record sounds for one full cardiac cycle [3-4]. These sounds result from vibrations created by the heart valve's closures. Sound S1 is produced when the atrioventricular valves close at the beginning of systole and the S2 when the aortic valve and pulmonary valve close at the end of systole. Phonocardiography helps to detect subaudible sounds and murmurs for one complete cardiac cycle making permanent logs of it. On the other hand, the stethoscope cannot always detect all such sounds or murmurs and also does not record the sound instances. The ability to capture the pattern in the sound is a subtle way of keeping a sanity check with the patient's heart condition. Other vitals collected include features like resting heartbeat, serum cholesterol, ST depression, number of major vessels coloured by fluoroscopy, etc. This list is exclusive but not exhaustive with the features. Serum cholesterol gives the border picture about the heart’s health. The contribution of this paper is summarized below:

1. To the best of author’s knowledge, it is the first study ensemble the ECG, PCG signal images and other vitals (with numerical data of the heart test report). The three types of inputs are processed individually by light weighted deep models to identify intermediate disease pattern representations for local disease prediction. Whereas, the global prediction is obtained by combining the local scores through score-level fusion.
2. PCG signals are first transformed into its short-term Fourier Transform (STFT), and their Mel spectrograms are fed to CNNs for disease prediction. Further, another
approach employed for PCG analysis is the time series analysis, where each sound file is passed through CNN plus Bi-LSTMs, to create a sequential model for prediction.
3. In order to combine three lightweight models for global prediction, the datasets containing ECG, PCG, and numerical data reports must refer to common patients. However, due to unavailability of such dataset, we develop a pseudo database synchronizing samples in all individual datasets to train the ensemble model.

This paper is organized as: Section II discusses the existing methods for diagnosis of CADs. Section III demonstrates the proposed ensemble method to detect the coronary disease based on ECG, PCG signals, and numerical data. Further, section IV demonstrates the benchmark datasets incorporated for the assessment of the method introduced in this paper. It also illustrates the experimental setup and results, along with the discussion on the observations and robustness of the model. Finally, section V summarizes the entire work and remarks.

## Proposed Methodology

Considering the correlation between the ECG and PCG signals for diagnosing CAD, this paper introduces an ensemble method incorporating both ECG and PCG images along with other vitals (numerical data) for CAD detection. The integrated use of ECG and PCG signals can conquer the drawback of using either of the two and thus perform more accurate CAD detection. In order to discriminate CAD and non-CAD subjects, individual deep learning based models are designed to identify disease patterns in ECG, PCG signals, and other vitals (with numerical values). Each model constitutes discriminatory feature vectors from the given input type (signal, image, or numerical data) and classify it as either CAD or non-CAD. However, in order to combine the strength of ECG and PCG signals, this work employs score-level fusion of the individual model’s outcomes. The fusion is performed at the classifier’s outcome, i.e., the individual output scores are combined through score-level fusion to obtain the final outcome. In other words, the proposed method employs deep learning with fusion methodology to discriminate CAD and non-CAD classes. The entire methodology can be better explained with the help of a flowchart.

## A. ECG Image Classification model

The ECG images are input to a lightweight CNN model to identify the hidden unique patterns of coronary diseases and classify them in CAD and non-CAD categories. ECG pictures are passed under complex convolutions to get the middle portrayals. These portrayals are then used to foresee the coronary illness of possible patients. The ECG dataset passed through a series of feature engineering to obtain the final representation of the data, to input to the model. Initially, images are converted from RGB to Grayscale, then cropped to exclude irrelevant features such as patient’s name, hospital’s name, date, time, etc. Next, a lightweight CNN is incorporated as the ECG classifier since the current feed forward neural organization isn’t appropriate for image grouping as there is an outstanding expansion in the quantity of free boundaries. This is due to the fact that the crude picture is straightforwardly managed disregarding the geography of the picture. With the approach of the CNN model, connection of spatially neighbouring pixels can be removed by applying a nonlinear channel. It is feasible to extricate different nearby elements of the picture by applying different channels. The motivation behind employing 2D CNN on the ECG picture structure instead of the ECG signal is that 2D convolutional and pooling layers are more appropriate for sifting the spatial territory of the ECG pictures [34]. In order to optimize the proposed CNN model, the ReLU activation is chosen over Leaky ReLU and Exponential LU as it exhibits better ECG characterization. Further, to minimize training and validation losses, there are several well-known optimizer functions exist such as Adam, Adagrad, and Adadelta. However, in our experiments, the optimal point reaches the earliest with Adam optimizer. Consequently, we used the Adam optimizer function with an initial learning rate of 0.01

## B.  PCG Signal Classification

The PCG signal waveforms in the given dataset are available in ‘.wav’ format. To ensure the compatibility of the dataset’s schema to the proposed model’s input schema, these waveforms are converted into 1D-tensors of fixed length. Although, inferring this length-constant(ℓ) for tensors was one of the crucial tasks. This arbitrary constant (ℓ) is chosen by considering the average length, the minimum length, and the maximum length of waveforms. As shown in Fig. 4, this chosen value ℓ, helps mitigating the loss of information from any given waveforms. By observing and plotting random data samples and considering the aforementioned parameters, the value of ℓ has chosen to be four thousand. After rigorous testing of multiple model architectures, two models were short-listed for classification. Further, best performing model is picked for final evaluation. Converting Waveforms to Spectrograms: The dataset's waveforms are represented in the time domain. In the proposed
method, waveforms are converted from the time-domain signals into the short-framed signals, also known as timefrequency-domain signals by evaluating the short-time Fourier transform to convert the waveforms to spectrograms , which can be represented as 2D images and physically signifies frequency changes over time.STFT splits
the signal into windows of time and runs a Fourier transform on each window, preserving some time information, and returning a 2D tensor image.

### Conv1D with Bi-LSTMs: RNN is a series of neural networks dedicated to processing sequence data. Several applications use RNN architecture, such as gated recurrent unit (GRU) and long short-term memory (LSTM), has been found to deliver the most advanced performance [38], including machine translation, speech recognition, and image captioning. Heart sound signal is a time series data with strong time correlation; therefore, it can be managed appropriately by RNN. In reality, these are proven to be incredibly useful, and they're frequently employed to classify heart sounds. RNN accepts input in the form of a one-dimensional heart sound signal x(t) =
(x1, XT) and, at the current time t, the hidden information or memory of the computing network, ht, in the use of RNN-based heart sound analysis methods. Use ht1 as the previous state and xt as the input signal. The output vector is projected onto the probability corresponding to the number of heart sound categories using the SoftMax function.

## Analysis of other Vitals:

In the third analysis, we used the patient's vital signs, such as heartbeat, serum cholesterol, low ST pressure, the number of main blood vessels colored by fluoroscopy, age, height, systolic and diastolic blood pressure, etc., to predict CAD. In addition, the main contribution of this research shows an effective method for predicting the risk of cardiac arrest by analyzing various benchmark datasets.

## Ensemble Model Results:

The proposed CAD detection approach combines the above three models to build an ensemble model through score-level fusion. The weight values in (6) are identified using the procedure specified in the subsection-III(D). The supervised classifier predicting the final result, uses a simple technique which assigns weightage to separate models according to its individual recall score performance. Further, probabilities of individual models are multiplied with this assigned weightage
score and then is summer across respective classes. Next, the weighted addition is normalized by taking the average and then final results are predicted from this normalized score.  

## Pseudo Sampling Technique for Ensemble Learning:

Due to the limitations on availability of all three types of data corresponding to a single patient, synthesis of pseudo database from existing database was necessary in order to accomplish goal of this paper. In this method, random samples for both classes were taken from all three separate databases available and this single unit was declared as the data corresponding to a single patient according to the appropriate label. By creating this pseudo database, it was inferred that, ensemble of all three models yields optimized recall score. This process was made iterative to generate minimal amount of data required for testing. The CAD detection performance of the ensemble model is demonstrated in Table VI, where the recall score is 0.95, which is higher than the individual performances of all three models proposed in this paper. The confusion matrix return by the ensemble model.

## CONCLUSION

The proposed method presented an exhaustive overview of the techniques, which can be employed for detecting CADs via a deep learning based approach. Automated analysis and classification of PCG signals and ECG images without the presence of any subject matter expert (SME) is a crucial task. Thus, developing reliable and robust systems, susceptible to noise, becomes essential. The classification of heartbeats has always been the main research area, with the purpose of distinguishing between normal and abnormal heart sounds. Varieties in beat sufficiency, recurrence(frequency) and span make PCG a complicated sign for programmed examination. This paper outlines the current state-of-the-art technology on the subject. ECG images, however, are in a standard format but require SMEs to infer from a report. This work attempts to neutralize the need for SMEs to infer from an ECG report. Although, these proposed methods come with additional computing costs and algorithmic complexities. Consequently, primary features like amplitude, pulse rate, diagnosis of SME from the medical domain continue to sustain. The proposed methods are enhanced with the help of machine learning to supplement existing methods and provide better classification performance.



## Results

 <img  align="center" src="https://github.com/Vishesht27/See-thru-Heart/blob/main/results_visualization/res_1.jpg">
 
 <img  align="center" src="https://github.com/Vishesht27/See-thru-Heart/blob/main/results_visualization/res_2.jpg">

 <img  align="center" src="https://github.com/Vishesht27/See-thru-Heart/blob/main/results_visualization/res_3.jpg">
 
  <img  align="center" src="https://github.com/Vishesht27/See-thru-Heart/blob/main/results_visualization/res_4.jpg">

## Contributors

## Build Process

### Tech-Stack-Involved

<div style="display: flex;justify-content: center;">
<img height="64px" width="auto" src="https://www.vectorlogo.zone/logos/tensorflow/tensorflow-icon.svg">
  <img height="64px" width="auto" src="https://upload.wikimedia.org/wikipedia/commons/thumb/a/ae/Keras_logo.svg/768px-Keras_logo.svg.png">

