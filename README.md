# Covid19_xray_detection

## About the Project
The project aims at building a robust CNN Architecture which is able to classify a given chest xray image into one of the three classes:
1. Covid19 
2. Other Pneumonia
3. Non Pneumonia

This project was carried out as a part of the project work of the course Deep Learning Foundations and Applications (AI61002), IIT Kharagpur under Prof. Debdoot Sheet sir.

## Model Architecture
The model used is relatively light weight.
Weights of pretrained Resnet18(trained on ImageNet dataset) downloaded from Pytorch were used. These weights were further tuned during training.

#### Layer 1
  Pretrained ResNet18(Pytorch, trained on ImageNet dataset)
  Input Dimensions:(channels = 3(RGB), 224, 224)
  Output Dimensions: (1000)
#### Layer 2
  Fully Connected Layer 1 (in=1000, out=200)
#### Layer 3
  Fully Connected Layer 2 (in=200, out=40)
#### Layer 4
  Fully Connected Layer 3 (in=40, out=10)
#### Layer 5
  Fully Connected Layer 4 (in=10, out=3)

A ReLU is placed after each Fully Connected Layer except the last layer which has a softmax.
Total Parameters = 11907815

## Dataset
  The dataset used in training this model has been obtained from sources mentioned in the References section(1-7).
## Preprocessing
  All images were converted to dimensions of (224x224) and into 3 channel RGB images.
  Data augmentation was carried out using techniques like colorjitter, random rotations (<15 degrees) and normalisation.
## Training
  This model was trained on Google Colab using the provided GPUs.
  Due to hardware limitations, the training was done in an unorthodox fashion. The hardware available could handle only a       dataset of the size of 500 images for training hence, at regular intervals the train loader had to be deleted and a new       train loader was loaded with a fresh set of 500 images from the dataset.
  Framework : Pytorch
  Learning rate = 1e-3
  Optimizer : Adam optimizer

## References
[1] https://twitter.com/ChestImaging/status/1243928581983670272

[2] https://www.sirm.org/category/senza-categoria/covid-19/

[3] Irvin, Jeremy, Pranav Rajpurkar, Michael Ko, Yifan Yu, Silviana Ciurea-Ilcus, Chris Chute, Henrik
    Marklund et al. "Chexpert: A large chest radiograph dataset with uncertainty labels and expert comparison."
    In Proceedings of the AAAI Conference on Artificial Intelligence, vol. 33, pp. 590-597. 2019. (link:
    https://stanfordmlgroup.github.io/competitions/chexpert/ )
    
[4] Radiological Society of North America, RSNA https://www.kaggle.com/c/rsna-pneumonia-detection-challenge
    pneumonia detection challenge.
    
[5] Joseph Paul Cohen and Paul Morrison and Lan Dao, “COVID-19 image data collection”,
    arXiv:2003.11597, 2020 https://github.com/ieee8023/covid-chestxray-dataset.
    
[6] Linda Wang, Alexander Wong, Zhong Qiu Lin, James Lee, Paul McInnis, Audrey Chung, Matt Ross,
    Blake VanBerlo, Ashkan Ebadi, “FIgure 1 COVID-19 Chest X-ray Dataset Initiative”,
    https://github.com/agchung/Figure1-COVID-chestxray-dataset
    
[7] Kong, Weifang, and Prachi P. Agarwal. "Chest imaging appearance of COVID-19 infection."
    Radiology: Cardiothoracic Imaging 2, no. 1 (2020): e200028.
    https://pubs.rsna.org/doi/full/10.1148/ryct.2020200028
    
[8] Deep Residual Learning for Image Recognition, Kaiming He and Xiangyu Zhang and Shaoqing Ren and Jian Sun, 2015
