# Crack-Detection-and-Segmentation-Dataset-for-UAV-Inspection
Here I have summarized different crack datasets and constructed a benchmark dataset for crack detection and segmentation. And this is the dataset which can be utilized for both crack detection and segmentation and it will be beneficial for the further research in this field.
# Contents
 - [Overview](#Overview)
 - [Dataset](#Dataset)
 - [Results](#Results)
 - [Citation](#Citation)
 
 # Overview
 Concrete structures such as bridge play an important role in ground transportation networks. While it is very labor intensive and dangerous for human to do the crack inspection. However, traditionally, concrete structure inspections are conducted manually by human operators with heavy and expensive mechanical equipment. It is logistically challenging, costly, and dangerous, especially when inspecting the substructure and superstructure in harsh environments that are hard and dangerous to be accessed by human operators. Therefore, it is very meaningful and significant for us to develop a fully autonomous intelligent unmanned aerial system for inspecting large-scale concrete structures and detecting the defects such as cracks. Most importantly, a significant module for UAV intelligent inspection system is to develop computer vision algorithms for processing images captured and detecting cracks and structural damages.
 
 #Dataset
 We have summarized different crack detection and segmentation dataset and established a benchmark dataset. The link is as follow and feel free to download it.
 When your are doing training based on the provided dataset, feel free to do the pre-processing such as cropping, resizing, rotating, normalizing and fliping to preprocess and enhance the dataset as the requirement.
 
#Results
Some results using feature pyrimid based convolutional neural networks to do segmentation are shown below. The Guided fiter was utilized to do post processing.

#Citation
Please cite the following papers if you are using these datasets.

CRACK500:
>@inproceedings{zhang2016road,
  title={Road crack detection using deep convolutional neural network},
  author={Zhang, Lei and Yang, Fan and Zhang, Yimin Daniel and Zhu, Ying Julie},
  booktitle={Image Processing (ICIP), 2016 IEEE International Conference on},
  pages={3708--3712},
  year={2016},
  organization={IEEE}
}' .

>@article{yang2019feature,
  title={Feature Pyramid and Hierarchical Boosting Network for Pavement Crack Detection},
  author={Yang, Fan and Zhang, Lei and Yu, Sijia and Prokhorov, Danil and Mei, Xue and Ling, Haibin},
  journal={arXiv preprint arXiv:1901.06340},
  year={2019}
}

GAPs384: 
>@inproceedings{eisenbach2017how,
  title={How to Get Pavement Distress Detection Ready for Deep Learning? A Systematic Approach.},
  author={Eisenbach, Markus and Stricker, Ronny and Seichter, Daniel and Amende, Karl and Debes, Klaus
          and Sesselmann, Maximilian and Ebersbach, Dirk and Stoeckert, Ulrike
          and Gross, Horst-Michael},
  booktitle={International Joint Conference on Neural Networks (IJCNN)},
  pages={2039--2047},
  year={2017}
}

CFD: 
>@article{shi2016automatic,
  title={Automatic road crack detection using random structured forests},
  author={Shi, Yong and Cui, Limeng and Qi, Zhiquan and Meng, Fan and Chen, Zhensong},
  journal={IEEE Transactions on Intelligent Transportation Systems},
  volume={17},
  number={12},
  pages={3434--3445},
  year={2016},
  publisher={IEEE}
}

AEL: 
>@article{amhaz2016automatic,
  title={Automatic Crack Detection on Two-Dimensional Pavement Images: An Algorithm Based on Minimal Path Selection.},
  author={Amhaz, Rabih and Chambon, Sylvie and Idier, J{\'e}r{\^o}me and Baltazart, Vincent}
}

cracktree200: 
>@article{zou2012cracktree,
  title={CrackTree: Automatic crack detection from pavement images},
  author={Zou, Qin and Cao, Yu and Li, Qingquan and Mao, Qingzhou and Wang, Song},
  journal={Pattern Recognition Letters},
  volume={33},
  number={3},
  pages={227--238},
  year={2012},
  publisher={Elsevier}
}

@article{shi2016automatic,
  title={Automatic road crack detection using random structured forests},
  author={Shi, Yong and Cui, Limeng and Qi, Zhiquan and Meng, Fan and Chen, Zhensong},
  journal={IEEE Transactions on Intelligent Transportation Systems},
  volume={17},
  number={12},
  pages={3434--3445},
  year={2016},
  publisher={IEEE}
}

@inproceedings{cui2015pavement,
  title={Pavement Distress Detection Using Random Decision Forests},
  author={Cui, Limeng and Qi, Zhiquan and Chen, Zhensong and Meng, Fan and Shi, Yong},
  booktitle={International Conference on Data Science},
  pages={95--102},
  year={2015},
  organization={Springer}
}

@article{liu2019deepcrack,
  title={DeepCrack: A Deep Hierarchical Feature Learning Architecture for Crack Segmentation},
  author={Liu, Yahui and Yao, Jian and Lu, Xiaohu and Xie, Renping and Li, Li},
  journal={Neurocomputing},
  volume={338},
  pages={139--153},
  year={2019},
  doi={10.1016/j.neucom.2019.01.036}
}

@inproceedings{yang2017deep,
  title={Deep concrete inspection using unmanned aerial vehicle towards cssc database},
  author={Yang, Liang and Li, Bing and Li, Wei and Liu, Zhaoming and Yang, Guoyong and Xiao, Jizhong},
  booktitle={Proceedings of the IEEE/RSJ International Conference on Intelligent Robots and Systems},
  pages={24--8},
  year={2017}
}
