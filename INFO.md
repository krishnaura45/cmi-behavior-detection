# Kaggle Competition ~ `CMI - Detect Behavior with Sensor Data`

## Overview

Can you use movement, temperature, and proximity sensor data to differentiate between body-focused repetitive behaviors (BFRBs), like hair pulling, from non-BFRB everyday gestures, like adjusting glasses? The goal of this competition is to develop a predictive model that distinguishes BFRB-like and non-BFRB-like activity using data from a variety of sensors collected via a wrist-worn device. Successfully disentangling these behaviors will improve the design and accuracy of wearable BFRB-detection devices, which are relevant to a wide range of mental illnesses, ultimately strengthening the tools available to support their treatment.

## Description

**Body-focused repetitive behaviors** (BFRBs), such as hair pulling, skin picking, and nail biting, are self-directed habits involving repetitive actions that, when frequent or intense, can cause physical harm and psychosocial challenges. These behaviors are commonly seen in anxiety disorders and obsessive-compulsive disorder (OCD), thus representing key indicators of mental health challenges.

To investigate BFRBs, the Child Mind Institute has developed a wrist-worn device, Helios, designed to detect these behaviors. While many commercially available devices contain Inertial Measurement Units (IMUs) to measure rotation and motion, the Helios watch integrates additional sensors, including 5 thermopiles (for detecting body heat) and 5 time-of-flight sensors (for detecting proximity). See the figure to the right for the placement of these sensors on the Helios device.

We conducted a research study to test the added value of these additional sensors for detecting BFRB-like movements. In the study, participants performed series of repeated gestures while wearing the Helios device:

They began a transition from “rest” position and moved their hand to the appropriate location (Transition);

They followed this with a short pause wherein they did nothing (Pause); and

Finally they performed a gesture from either the BFRB-like or non-BFRB-like category of movements (Gesture; see Table below).

Each participant performed 18 unique gestures (8 BFRB-like gestures and 10 non-BFRB-like gestures) in at least 1 of 4 different body-positions (sitting, sitting leaning forward with their non-dominant arm resting on their leg, lying on their back, and lying on their side). These gestures are detailed in the table below, along with a video of the gesture.

BFRB-Like Gesture (Target Gesture)	Video Example Links
Above ear - Pull hair	Sitting
Forehead - Pull hairline	Sitting leaning forward
Forehead - Scratch	Sitting
Eyebrow - Pull hair	Sitting
Eyelash - Pull hair	Sitting
Neck - Pinch skin	Sitting
Neck - Scratch	Sitting
Cheek - Pinch skin	Sitting,
Sitting leaning forward,
Lying on back,
Lying on side


Non-BFRB-Like Gesture (Non-Target Gesture)	Video Example Links
Drink from bottle/cup	Sitting
Glasses on/off	Sitting
Pull air toward your face	Sitting
Pinch knee/leg skin	Sitting leaning forward
Scratch knee/leg skin	Sitting leaning forward
Write name on leg	Sitting leaning forward
Text on phone	Sitting
Feel around in tray and pull out an object	Sitting
Write name in air	Sitting
Wave hello	Sitting

This competition challenges you to develop a predictive model capable of distinguishing (1) BFRB-like gestures from non-BFRB-like gestures and (2) the specific type of BFRB-like gesture. Critically, when your model is evaluated, half of the test set will include only data from the IMU, while the other half will include all of the sensors on the Helios device (IMU, thermopiles, and time-of-flight sensors).

Your solutions will have direct real-world impact, as the insights gained will inform design decisions about sensor selection — specifically whether the added expense and complexity of thermopile and time-of-flight sensors is justified by significant improvements in BFRB detection accuracy compared to an IMU alone. By helping us determine the added value of these thermopiles and time-of-flight sensors, your work will guide the development of better tools for detection and treatment of BFRBs.

Relevant articles:
[Garey, J. (2025). What Is Excoriation, or Skin-Picking? Child Mind Institute](https://childmind.org/article/excoriation-or-skin-picking/).

[Martinelli, K. (2025). What is Trichotillomania? Child Mind Institute](https://childmind.org/article/what-is-trichotillomania/).

## Evaluation

The evaluation metric for this contest is a version of macro F1 that equally weights two components:

Binary F1 on whether the gesture is one of the target or non-target types.
Macro F1 on gesture, where all non-target sequences are collapsed into a single non_target class
The final score is the average of the binary F1 and the macro F1 scores.

If your submission includes a gesture value not found in the train set your submission will trigger an error.

### Submission File

You must submit to this competition using the provided evaluation API, which ensures that models perform inference on a single sequence at a time. For each sequence_id in the test set, you must predict the corresponding gesture.

## Timeline

May 29, 2025 - Start Date.

August 26, 2025 - Entry Deadline. You must accept the competition rules before this date in order to compete.

August 26, 2025 - Team Merger Deadline. This is the last day participants may join or merge teams.

September 2, 2025 - Final Submission Deadline.

All deadlines are at 11:59 PM UTC on the corresponding day unless otherwise noted. The competition organizers reserve the right to update the contest timeline if they deem it necessary.

## Acknowledgements

The data used for this competition was provided in collaboration with the Healthy Brain Network, a landmark mental health study based in New York City that will help children around the world. In the Healthy Brain Network, families, community leaders, and supporters are partnering with the Child Mind Institute to unlock the secrets of the developing brain. Additional study participants were recruited from Child Mind Institute’s staff and community, and we are grateful for their collaboration. In addition to the generous support provided by the Kaggle team, financial support has been provided by the California Department of Health Care Services (DHCS) as part of the Children and Youth Behavioral Health Initiative (CYBHI).

DHCS DHCS

## Code Requirements

Submissions to this competition must be made through Notebooks. In order for the "Submit" button to be active after a commit, the following conditions must be met:

CPU Notebook <= 9 hours run-time
GPU Notebook <= 9 hours run-time
Internet access disabled
Freely & publicly available external data is allowed, including pre-trained models
Please see the Code Competition FAQ for more information on how to submit. And review the code debugging doc if you are encountering submission errors.

## About the Child Mind Institute and the Healthy Brain Network

DHCS The Child Mind Institute (CMI) is the leading independent nonprofit in children’s mental health providing gold-standard, evidence-based care, delivering educational resources to millions of families each year, training educators in underserved communities, and developing open science initiatives and tomorrow’s breakthrough treatments. The Healthy Brain Network (HBN) is a community-based research initiative of the Child Mind Institute. We provide no-cost, study-related mental health and learning evaluations to children ages 5–21 and connect families with community resources. We are collecting the information needed to find brain and body characteristics that are associated with mental health and learning disorders. The Healthy Brain Network stores and openly shares de-identified data about psychiatric, behavioral, cognitive, and lifestyle (e.g., fitness, diet) phenotypes, as well as multimodal brain imaging (MRI), electroencephalography (EEG), digital voice and video recordings, genetics, and actigraphy.

## Citation

Laura Newman, David LoBue, Arianna Zuanazzi, Florian Rupprecht, Luke Mears, Roxanne McAdams, Erin Brown, Yanyi Wang, Camilla Strauss, Arno Klein, Lauren Hendrix, Maki Koyama, Josh To, Curt White, Yuki Kotani, Michelle Freund, Michael Milham, Gregory Kiar, Martyna Plomecka, Sohier Dane, and Maggie Demkin. [CMI - Detect Behavior with Sensor Data](https://kaggle.com/competitions/cmi-detect-behavior-with-sensor-data), 2025. Kaggle.
