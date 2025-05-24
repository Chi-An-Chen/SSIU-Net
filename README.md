# SSIU-Net : A State Space-Inspired U-Net with Linear Attention for Breast Tumor Image Segmentation  
<div align="center">

  [[`Paper`](assets/paper.png)]
  
  This is the official PyTorch implementation of SSIU-Net: a State Space-Inspired U-Net that integrates a State Space Model (SSM) with a Mamba-Inspired Linear Attention Block (MILA) for breast tumor image segmentation.  
  
  <img src="assets/model.png" width="50%"><br>
  Fig 1. Overall Architecture of LViT-CB model.  
  
  <img src="assets/blk.png" width="50%"><br>
  Fig 2. Architecture Overview of the Competition Block (Top) and Competition Module (Bottom).  

</div>

## Environments  
```
conda env create -f environment.yaml
```
```
conda activate fingervein
```

## Prepare dataset for Training and Testing
Get a .pkl file contain data's path 
```
python 000_make_Dataset.py
```  

## Training
Start training the model 
```
python 001_train.py
```

## Testing 
Start evaluating the model
```
python 002_test.py
```

## Model's structure and details
```
python model_detail.py
```

# References： 
> * C.-H. Hsia, L.-Y. Ke, and S.-T. Chen, “Improved lightweight convolutional neural network for finger vein recognition system,” _Bioengineering_, vol. 10, no. 8, pp. 919, 2023.  
> [[GitHub]](https://github.com/liangying-Ke/ILCNN)
> * T.F. Zhang, et al., "Cas-vit: Convolutional additive self-attention vision transformers for efficient mobile applications," _ArXiv_ ,arXiv:2408.03703, 2024.  
> [[GitHub]](https://github.com/Tianfang-Zhang/CAS-ViT)
> * Z. Yang, et al., "Comprehensive Competition Mechanism in Palmprint Recognition," _IEEE Transactions on Information Forensics and Security_, vol. 18, pp. 5160-5170, 2023.  
>[[GitHub]](https://github.com/Zi-YuanYang/CCNet)

