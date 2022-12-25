<h1 align="center">Sea-thru-Heart</h1> <br>


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

