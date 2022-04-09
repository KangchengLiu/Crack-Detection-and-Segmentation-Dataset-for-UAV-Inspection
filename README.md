# Crack-Detection-and-Segmentation-Dataset-for-UAV-Inspection
Here I have summarized different crack datasets and constructed a benchmark dataset for crack detection and segmentation. And this is the dataset which can be utilized for both crack detection and segmentation and it will be beneficial for the further research in this field.
# Contents
 - [Overview](#Overview)
 - [Dataset Download](#Dataset)
 - [Results](#Results)
 - [Citation](#Citation)
 
 # Overview
 Concrete structures such as bridge play an important role in ground transportation networks. While it is very labor intensive and dangerous for human to do the crack inspection. However, traditionally, concrete structure inspections are conducted manually by human operators with heavy and expensive mechanical equipment. It is logistically challenging, costly, and dangerous, especially when inspecting the substructure and superstructure in harsh environments that are hard and dangerous to be accessed by human operators. Therefore, it is very meaningful and significant for us to develop a fully autonomous intelligent unmanned aerial system for inspecting large-scale concrete structures and detecting the defects such as cracks. Most importantly, a significant module for UAV intelligent inspection system is to develop computer vision algorithms for processing images captured and detecting cracks and structural damages.
 
 # Dataset Download
 We have summarized different crack detection and segmentation datasets and established a benchmark dataset. The link is as follow and feel free to download it.
 When your are doing training based on the provided dataset, feel free to do the pre-processing such as cropping, resizing, rotating, normalizing and fliping to preprocess and enhance the dataset as the requirement.
You can download the dataset from [the link](https://drive.google.com/open?id=1RMf0GYXn7Mq1s9STGFG5iByavTr05SjF).
# Results
The randomly selected crack detection results of sliding window approach are shown as follows:
![Represent](./images/sw2.png)
Some results using feature pyrimid based convolutional neural networks to do segmentation are shown below. The Guided Filter is utilized to do post processing.
![Represent](./images/crack_image.png)
Following are some typical detection results in some challenging circumstances with various cracks and noises.
![Represent](./images/crack_typical_results.png)
