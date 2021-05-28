# AdderNet: Do We Really Need Multiplications in Deep Learning?
This code is a demo of CVPR 2020 paper [AdderNet: Do We Really Need Multiplications in Deep Learning?](https://openaccess.thecvf.com/content_CVPR_2020/papers/Chen_AdderNet_Do_We_Really_Need_Multiplications_in_Deep_Learning_CVPR_2020_paper.pdf) 

We present adder networks (AdderNets) to trade massive multiplications in deep neural networks, especially convolutional neural networks (CNNs), for much cheaper additions to reduce computation costs. In AdderNets, we take the L1-norm distance between filters and input feature as the output response. As a result, the proposed AdderNets can achieve 74.9% Top-1 accuracy 91.7% Top-5 accuracy using ResNet-50 on the ImageNet dataset without any multiplication in convolution layer.


### UPDATE: The training code is released in 6/28.

Run `python main.py` to train on CIFAR-10. 

<p align="center">
<img src="figures/visualization.png" width="800">
</p>

### UPDATE: Model Zoo about AdderNets are released in 11/27.

Classification results on CIFAR-10 and CIFAR-100 datasets.

| Model     | Method           | CIFAR-10 | CIFAR-100 |
| --------- | ---------------- | -------- | --------- |
| VGG-small | ANN [1]          | 93.72%   | 74.58%    |
|           | PKKD ANN [2]     | 95.03%   | 76.94%    |
|           |                  |          |           |
| ResNet-20 | ANN              | 92.02%   | 67.60%    |
|           | PKKD ANN         | 92.96%   | 69.93%    |
|           | ShiftAddNet* [3] | 89.32%(160epoch)   | -         |
|           |                  |          |           |
| ResNet-32 | ANN              | 93.01%   | 69.17%    |
|           | PKKD ANN         | 93.62%   | 72.41%    |

Classification results on ImageNet dataset.

| Model     | Method       | Top-1 Acc | Top-5 Acc |
| --------- | ------------ | --------- | --------- |
| ResNet-18 | CNN          | 69.8%     | 89.1%     |
|           | ANN [1]      | 67.0%     | 87.6%     |
|           | PKKD ANN [2] | 68.8%     | 88.6%     |
|           |              |           |           |
| ResNet-50 | CNN          | 76.2%     | 92.9%     |
|           | ANN          | 74.9%     | 91.7%     |
|           | PKKD ANN     | 76.8%     | 93.3%     |

Super-Resolution results on several SR datasets.

| Scale | Model | Method  | Set5 (PSNR/SSIM) | Set14 (PSNR/SSIM) | B100 (PSNR/SSIM) | Urban100 (PSNR/SSIM) |
| ----- | ----- | ------- | ---------------- | ----------------- | ---------------- | -------------------- |
| ×2    | VDSR  | CNN     | 37.53/0.9587     | 33.03/0.9124      | 31.90/0.8960     | 30.76/0.9140         |
|       |       | ANN [4] | 37.37/0.9575     | 32.91/0.9112      | 31.82/0.8947     | 30.48/0.9099         |
|       | EDSR  | CNN     | 38.11/0.9601     | 33.92/0.9195      | 32.32/0.9013     | 32.93/0.9351         |
|       |       | ANN     | 37.92/0.9589     | 33.82/0.9183      | 32.23/0.9000     | 32.63/0.9309         |
| ×3    | VDSR  | CNN     | 33.66/0.9213     | 29.77/0.8314      | 28.82/0.7976     | 27.14/0.8279         |
|       |       | ANN     | 33.47/0.9151     | 29.62/0.8276      | 28.72/0.7953     | 26.95/0.8189         |
|       | EDSR  | CNN     | 34.65/0.9282     | 30.52/0.8462      | 29.25/0.8093     | 28.80/0.8653         |
|       |       | ANN     | 34.35/0.9212     | 30.33/0.8420      | 29.13/0.8068     | 28.54/0.8555         |
| ×4    | VDSR  | CNN     | 31.35/0.8838     | 28.01/0.7674      | 27.29/0.7251     | 25.18/0.7524         |
|       |       | ANN     | 31.27/0.8762     | 27.93/0.7630      | 27.25/0.7229     | 25.09/0.7445         |
|       | EDSR  | CNN     | 32.46/0.8968     | 28.80/0.7876      | 27.71/0.7420     | 26.64/0.8033         |
|       |       | ANN     | 32.13/0.8864     | 28.57/0.7800      | 27.58/0.7368     | 26.33/0.7874         |

*ShiftAddNet [3] used different training setting.

[1]  **AdderNet: Do We Really Need Multiplications in Deep Learning?**  *Hanting Chen, Yunhe Wang, Chunjing Xu, Boxin Shi, Chao Xu, Qi Tian, Chang Xu.* **CVPR, 2020. (Oral)**

[2] **Kernel Based Progressive Distillation for Adder Neural Networks**. *Yixing Xu, Chang Xu, Xinghao Chen, Wei Zhang, Chunjing XU, Yunhe Wang.* **NeurIPS, 2020. (Spotlight)**

[3] **ShiftAddNet: A Hardware-Inspired Deep Network.** *Haoran You, Xiaohan Chen, Yongan Zhang, Chaojian Li, Sicheng Li, Zihao Liu, Zhangyang Wang, Yingyan Lin.* **NeurIPS, 2020.**

[4] **AdderSR: Towards Energy Efficient Image Super-Resolution**. *Dehua Song, Yunhe Wang, Hanting Chen, Chang Xu, Chunjing Xu, Dacheng Tao*. **Arxiv, 2020.** 

## Requirements

- python 3
- pytorch >= 1.1.0
- torchvision

### Preparation
You can follow [pytorch/examples](https://github.com/pytorch/examples/tree/master/imagenet) to prepare the ImageNet data.

The pretrained models are available in [google drive](https://drive.google.com/drive/folders/11ZdIst5Vwqx9Y5zHhirfaI94_7RhcBZH?usp=sharing) or [baidu cloud](https://pan.baidu.com/s/1pkaWhhKVoWPv-MCUjvxzCw) (access code:126b)

### Usage
Run `python main.py` to train on CIFAR-10. 

Run `python test.py --data_dir 'path/to/imagenet_root/'` to evaluate on ImageNet `val` set. You will achieve 74.9% Top accuracy and 91.7% Top-5 accuracy on the ImageNet dataset using ResNet-50.

Run `python test.py --dataset cifar10 --model_dir models/ResNet20-AdderNet.pth --data_dir 'path/to/cifar10_root/'` to evaluate on CIFAR-10. You will achieve 91.8% accuracy on the CIFAR-10 dataset using ResNet-20.

The inference and training of AdderNets is slow since the adder filters is implemented without cuda acceleration. You can write [cuda](https://docs.nvidia.com/cuda/cuda-samples/index.html) to achieve higher inference speed. 

## Citation
	@article{AdderNet,
		title={AdderNet: Do We Really Need Multiplications in Deep Learning?},
		author={Chen, Hanting and Wang, Yunhe and Xu, Chunjing and Shi, Boxin and Xu, Chao and Tian, Qi and Xu, Chang},
		journal={CVPR},
		year={2020}
	}

### Contributing
We appreciate all contributions. If you are planning to contribute back bug-fixes, please do so without any further discussion.

If you plan to contribute new features, utility functions or extensions to the core, please first open an issue and discuss the feature with us. Sending a PR without discussion might end up resulting in a rejected PR, because we might be taking the core in a different direction than you might be aware of.
