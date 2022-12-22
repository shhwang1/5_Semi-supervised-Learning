# Semi-supervised-Learning Tutorial

## 목차

## 0. Overview of Semi-Supervised Learning

## 1. Hybrid/Holistic Methods
- ### 1-1. MixMatch
- ### 1-2. FixMatch
- ### 1-3. FlexMatch
___
## 2. Experimental Analysis
## 3. Conclusion
___
## 0. Overview of Semi-Supervised Learning
![image](https://blog.est.ai/assets/img/2020/1109/1.jpeg)

## - 준지도학습(Semi-supervised Learning)이란?

지도 학습(Supervised Learning)을 위해서 데이터에 레이블을 부여하는 '데이터 레이블링' 작업은 매우 많은 자원과 비용이 소모됩니다. 이러한 문제점을 타파하고자, 레이블링이 적용되지 않은, unlabeled 데이터를 활용하는 방법론들이 등장했는데요. 우리는 이를 비지도학습(Unsupervised Learning)이라고 합니다. 비지도학습은 기존의 지도학습과는 다르게, 정답(레이블)이 없는 데이터 간 유사한 특징을 찾아 비슷한 데이터끼리 묶는 것을 목표로 하는데, 우리는 이를 클러스터링(Clustering)이라고 합니다. 

이처럼 레이블링이 모두 적용된 데이터를 활용하는 "지도학습"과, 레이블링이 적용되지 않은 데이터를 바탕으로 클러스터링을 목표로 하는 "비지도학습" 그 중간의 방법론을 준지도학습(Semi-supervised Learning)이라고 합니다. 위의 사진을 보시게 되면 가운데 그림이 준지도학습을 나타내고 있는데요, 보시면 아시겠지만 색깔이 칠해져있는, 극히 일부의 레이블링이 적용된 데이터를 통해 우선적으로 클래스(레이블)간 경계를 1차적으로 형성합니다. 그 후, 레이블링이 적용되지 않은 비지도학습용 데이터를 활용해서 그 경계를 다듬는, 더 정확한 경계를 형성하도록 하는 방법론이 준지도학습이라고 생각하시면 됩니다.

준지도학습에는 Consistency Regularization 관점, Hybrid/Holistic Methods 관점 크게 두 가지로 나뉘는데, 본 튜토리얼에서는 Hybrid/Holistic Methods의 MixMatch, FixMatch, FlexMatch 세 모델을 from scratch 형식으로 코드화해보고자 합니다.
___

- ## 활용 데이터

본 튜토리얼에서 사용할 데이터셋은 이미지 데이터셋으로 유명한 CIFAR-10 데이터셋입니다. 이름에서처럼 총 10개의 Class가 존재하는 데이터셋이고, 클래스별로 각 6,000개의 이미지, 즉 총 60,000개의 샘플이 존재합니다. 샘플의 예시는 아래의 사진을 참고하세요.

![image](https://miro.medium.com/max/505/1*r8S5tF_6naagKOnlIcGXoQ.png)
___
# Hybrid/Holistic Methods
![image](https://user-images.githubusercontent.com/115224653/208087192-bf2db310-8df7-4b70-9571-ee8517059982.png)

본 튜토리얼에서 다룰 Hybrid/Holistic Methods는 위 세가지, MixMatch, FixMatch, FlexMatch인데요. 사진의 포켓몬처럼 세 방법론들은 서로의 단점을 보완하면서 서서히 진화한 방법론의 관계가 있습니다. 그렇다면, 가장 초기 방법론인 MixMatch부터 살펴보도록 하겠습니다.
___
## 1. MixMatch
![image](https://euphoria0-0.github.io/assets/img/posts/2021-01-08-Semi-Supervised-Learning-and-MixMatch/MixMatch-labelguessing.png)

MixMatch는 Augmentation 기법을 적극 활용하는 방법론으로, unlabeled data와 labeled data를 모두 활용하는데요, 위의 그림과 같이 MixMatch가 진행되는 순서를 요약하면 아래의 순서와 같습니다.

- #### 1. Labeled data인 X에 대해서 Augmentation을 적용합니다.
- #### 2. Unlabeled data인 X'에 대해서도 Augmentation을 적용합니다.
- #### 3. Unlabeled data X'에 대해 Augmentation을 적용한 결과들에 대한 평균값을 냅니다. -> "q_b_bar" 
- #### 4. q_b_bar에 대해서 temperature sharpening 과정을 통해 가장 높은 class의 확률 값이 다른 class의 확률 값에 비해 월등히 높도록 만들어 줍니다.
- #### 5. Mixup 방법을 통해 학습을 위한 데이터를 재생성합니다. - 'X_MixSet'
- #### 6. Augmentation이 적용된 X, X'을 Concatenate하여 데이터셋을 만들어줍니다. -> 'W set'
- #### 7. 구성된 W set을 이용해 Mixup을 적용하여 새로운 데이터셋 'U'를 만들어 줍니다.
- #### 8. 5번의 'X_MixSet'과 7번의 'U Set'을 이용해서 아래의 Loss function을 적용하여 학습합니다.

![image](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FuCt9K%2Fbtq2kqYoIcw%2Fak2hfWkGmkkjrmtSNCn6qk%2Fimg.png)
___
# Python Code
## 모델에 사용되는 하이퍼파라미터 정의 (argparse)
``` py
import argparse

# 실험에서 사용할 하이퍼파라미터 정의
def MixMatch_parser():
    parser = argparse.ArgumentParser(description="MixMatch from Scratch")
    
    parser.add_argument('--n-labeled', type=int, default=1024) # 사용할 Labeled data의 수를 정의
    parser.add_argument('--num-iter', type=int, default=1024) # 한 epoch당 포함되는 iteration의 수를 정의
    parser.add_argument('--alpha', type=float, default=0.75) # threshold를 위한 alpha값 정의
    parser.add_argument('--lambda-u', type=float, default=75) # lambda-u 값 정의
    parser.add_argument('--T', default=0.5, type=float) 
    parser.add_argument('--ema-decay', type=float, default=0.999) # exponential moving average 가중치값 조절

    parser.add_argument('--epochs', type=int, default=20) # Epoch 정의
    parser.add_argument('--batch-size', type=int, default=64) # 학습에 사용할 batch size 크기 정의
    parser.add_argument('--lr', type=float, default=0.002) # 학습률(learning rate) 조절

    return parser
```
MixMatch의 학습에서 사용될 하이퍼파라미터는 대략적으로 labeled data의 수, iteration, alpha, lambda-u, T, ema-decay 정도가 있습니다. 각각의 하이퍼파라미터는 아래의 코드에서 나올때마다 어떤 의미를 갖는 지 설명하겠습니다.

``` py
import argparse

# 실험에서 사용할 하이퍼파라미터 정의
def MixMatch_parser():
    parser = argparse.ArgumentParser(description="MixMatch from Scratch")
    
    parser.add_argument('--n-labeled', type=int, default=1024) # 사용할 Labeled data의 수를 정의
    parser.add_argument('--num-iter', type=int, default=1024) # 한 epoch당 포함되는 iteration의 수를 정의
    parser.add_argument('--alpha', type=float, default=0.75) # threshold를 위한 alpha값 정의
    parser.add_argument('--lambda-u', type=float, default=75) # lambda-u 값 정의
    parser.add_argument('--T', default=0.5, type=float) 
    parser.add_argument('--ema-decay', type=float, default=0.999) # exponential moving average 가중치값 조절

    parser.add_argument('--epochs', type=int, default=20) # Epoch 정의
    parser.add_argument('--batch-size', type=int, default=64) # 학습에 사용할 batch size 크기 정의
    parser.add_argument('--lr', type=float, default=0.002) # 학습률(learning rate) 조절

    return parser
```
MixMatch는 위에서 설명드린 바와 같이, unlabeled data와 labeled data에 augmentation을 적용하여 사용하기 때문에 unlabeled data와 labeled data를 구성하는 custom dataset 코드가 필요합니다. 

우선적으로 아래의 코드를 살펴보시면 Normalize 함수를 통해 전체적인 데이터에 대한 정규화 작업을 거쳤습니다. 

그리고 Labeled_Cifar10, Unlabeled_Cifar10 함수를 통해 각각 labeled data, unlabeled data를 지정하는 코드를 구성하였습니다. 이 때, indexing을 활용하여 특정 index는 labeled data로, 특정 index는 unlabeled data로 구성하도록 설계하였습니다.
```py
import torchvision
import numpy as np

from utils.datasplit import split_datasets
from utils.transform import Transform_Twice

# Image를 전처리 하기 위한 함수

### 데이터를 정규화 하기 위한 함수
def Normalize(x, m=(0.4914, 0.4822, 0.4465), std=(0.2471, 0.2345, 0.2616)):
        
    ##### x, m, std를 각각 array화
    x, m, std = [np.array(a, np.float32) for a in (x, m, std)] 

    ##### 데이터 정규화
    x -= m * 255 
    x *= 1.0/(255*std)
    return x

### 데이터를 (B, C, H, W)로 수정해주기 위한 함수 (from torchvision.transforms 내 ToTensor 와 동일한 함수)
def Transpose(x, source='NHWC', target='NCHW'):
    return x.transpose([source.index(d) for d in target])

### 특정 이미지에 동서남북 방향으로 4만큼 픽셀을 추가해주기 위한 학습
def pad(x, border=4):
    return np.pad(x, [(0, 0), (border, border), (border, border)], mode='reflect')

# Labeled data를 생성하는 함수
class Labeled_CIFAR10(torchvision.datasets.CIFAR10):
    
    def __init__(self, root, indices=None,
                train=True, transform=None,
                target_transform=None, download=False):
        
        super(Labeled_CIFAR10, self).__init__(root,
                                        train=train,
                                        transform=transform,
                                        target_transform=target_transform,
                                        download=download)

        if indices is not None:
            self.data = self.data[indices]
            self.targets = np.array(self.targets)[indices]
        
        self.data = Transpose(Normalize(self.data))
    
    def __getitem__(self, index):
        
        img, target = self.data[index], self.targets[index]
        
        if self.transform is not None:
            img = self.transform(img)
        
        if self.target_transform is not None:
            target = self.target_transform(target)
        
        return img, target


# Unlabeled data를 생성하는 함수
class Unlabeled_CIFAR10(Labeled_CIFAR10):
    '''
    Unlabeled data의 Label은 -1로 지정
    '''
    def __init__(self, root, indices, train=True, transform=None, target_transform=None, download=False):
        
        super(Unlabeled_CIFAR10, self).__init__(root, indices, train,
                                            transform=transform,
                                            target_transform=target_transform,
                                            download=download)
        
        self.targets = np.array([-1 for i in range(len(self.targets))])


# CIFAR10에 대하여 labeled, unlabeled, validation, test dataset 생성
def get_cifar10(data_dir: str, n_labeled: int,
                transform_train=None, transform_val=None,
                download=True):
    
    ### Torchvision에서 제공해주는 CIFAR10 dataset Download
    base_dataset = torchvision.datasets.CIFAR10(data_dir, train=True, download=download)
    
    ### labeled, unlabeled, validation data에 해당하는 index를 가져오기
    indice_labeled, indice_unlabeled, indice_val = split_datasets(base_dataset.targets, int(n_labeled/10)) ### n_labeled는 아래 MixMatch_argparser 함수에서 정의
    
    ### index를 기반으로 dataset을 생성
    train_labeled_set = Labeled_CIFAR10(data_dir, indice_labeled, train=True, transform=transform_train) 
    train_unlabeled_set = Unlabeled_CIFAR10(data_dir, indice_unlabeled, train=True, transform=Transform_Twice(transform_train))
    val_set = Labeled_CIFAR10(data_dir, indice_val, train=True, transform=transform_val, download=True) 
    test_set = Labeled_CIFAR10(data_dir, train=False, transform=transform_val, download=True) 

    return train_labeled_set, train_unlabeled_set, val_set, test_set
```

아래는 Augmentation 리스트 및 적용 코드입니다. 

RandomPadandCrop, RandomFlip, GaussianNoise의 방법이 존재하고, ToTensor의 경우는 추후에 활용될 때 설명드리겠습니다.

```py
import torch
import numpy as np
from dataloader.CIFAR10_Loader import pad

# Image를 Augmentation하기 위한 함수

### Image를 Padding 및 Crop적용
'''
1. object는 써도 되고 안써도 되는 것
2. assert는 오류를 유도하기 위함 (나중에 이렇게 해놓으면 디버깅이 편함) --> 여기선 적절한 데이터 인풋의 형태를 유도
'''
class RandomPadandCrop(object):
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size
    
    def __call__(self, x):
        x = pad(x, 4)
        
        old_h, old_w = x.shape[1: ]
        new_h, new_w = self.output_size
        
        top = np.random.randint(0, old_h-new_h)
        left = np.random.randint(0, old_w-new_w)
        
        x = x[:, top:top+new_h, left:left+new_w]
        return x
    
### RandomFlip하는 함수 정의
class RandomFlip(object):
    def __call__(self, x):
        if np.random.rand() < 0.5:
            x = x[:, :, ::-1]
        
        return x.copy()
    
### GaussianNoise를 추가하는 함수 정의
class GaussianNoise(object):
    def __call__(self, x):
        c, h, w = x.shape
        x += np.random.randn(c, h, w)*0.15
        return x

# Numpy를 Tensor로 변환하는 함수
class ToTensor(object):
    def __call__(self, x):
        x = torch.from_numpy(x)
        return x
```
모델의 구조입니다.

논문에서와 동일하게 wideResNet을 통해 구현하였습니다.
wideResNet의 코드 같은 경우 직접 구현하지는 않았고, 아래의 github 링크에서 발췌하여 사용하였습니다.

Link : https://github.com/xternalz/WideResNet-pytorch/blob/master/wideresnet.py
```py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0, activate_before_residual=False):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes, momentum=0.001)
        self.relu1 = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes, momentum=0.001)
        self.relu2 = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                               padding=0, bias=False) or None
        self.activate_before_residual = activate_before_residual
    def forward(self, x):
        if not self.equalInOut and self.activate_before_residual == True:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)

class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0, activate_before_residual=False):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, dropRate, activate_before_residual)
    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate, activate_before_residual):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, dropRate, activate_before_residual))
        return nn.Sequential(*layers)
    def forward(self, x):
        return self.layer(x)

class WideResNet(nn.Module):
    def __init__(self, num_classes, depth=28, widen_factor=2, dropRate=0.0):
        super(WideResNet, self).__init__()
        nChannels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
        assert((depth - 4) % 6 == 0)
        n = (depth - 4) / 6
        block = BasicBlock
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        # 1st block
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate, activate_before_residual=True)
        # 2nd block
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate)
        # 3rd block
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(nChannels[3], momentum=0.001)
        self.relu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.fc = nn.Linear(nChannels[3], num_classes)
        self.nChannels = nChannels[3]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.zero_()

    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.nChannels)
        return self.fc(out)
```
아래는 지수평활법(exponential moving average)을 통해 가중치를 update하는 코드이다.
WeightEMA를 하는 이유는 학습시간이 길어지는 것을 방지하고, Trivial Solution을 방지 및 과적합을 방지하기 위함입니다. 가중치를 업데이트 시 a(최근가중치)+(1-a)(이전가중치) 의 수식으로 진행됩니다.

요약 : ema_params_new = self.decay*ema_params_old + (1-self.decay)*params
```py
import copy
import torch
# WeightEMA로 Parameter를 Update하는 함수를 정의 (EMA=Exponential Moving Average)

class WeightEMA(object): 

    def __init__(self, model, ema_model, lr, alpha=0.999):
        self.model = model
        self.ema_model = ema_model

        self.alpha = alpha

        self.params = list(self.model.state_dict().items())
        self.ema_params = list(self.ema_model.state_dict().items())

        self.wd = 0.02 * lr

        for param, ema_param in zip(self.params, self.ema_params):
            param[1].data.copy_(ema_param[1].data)
    
    def step(self):
        inverse_alpha = 1.0 - self.alpha
        for param, ema_param in zip(self.params, self.ema_params):
            if ema_param[1].dtype == torch.float32:
                ema_param[1].mul_(self.alpha) # ema_params_new = self.alpha * ema_params_old
                ema_param[1].add_(param[1]*inverse_alpha) # ema_params_Double_new = (1-self.alpha)*params

                param[1].mul_(1-self.wd)
```

아래는 loss function에 대한 코드입니다. MixMatch 알고리즘에는 supervised loss와 semi-supervised loss가 존재하는데, supervised loss는 기본적으로 구현되어있는 CrossEntropy 함수를 활용하면 되기 때문에 해당 코드에는 별다른 구현 코드가 없습니다. 대신, Semi-supervised loss는 아래와 같이 구현되어 있습니다.

```py
import numpy as np
import torch

import torch.nn.functional as F


def linear_rampup(current, rampup_length):
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current/rampup_length, 0.0, 1.0)
        return float(current)


class Loss_Semisupervised(object):
    def __call__(self, args, outputs_x, target_x, outputs_u, targets_u, epoch):
        self.args = args
        probs_u = torch.softmax(outputs_u, dim=1)

        loss_x = -torch.mean(
            torch.sum(F.log_softmax(outputs_x, dim=1)*target_x, dim=1)
        )

        loss_u = torch.mean((probs_u-targets_u)**2)

        return loss_x, loss_u, self.args.lambda_u*linear_rampup(epoch, self.args.epochs)
```

마지막으로, 학습 코드입니다.

실험 결과 파일을 저장할 폴더를 지정합니다. 그 후 데이터에 대한 transformation을 적용하고, dataset을 학습용, 검증용, 테스트용으로 분리합니다. Semi-supervised Loss와 Supervised Loss을 위한 CrossEntropy를 설정하고, 모델을 학습하게 됩니다.

```py
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import transforms as transforms
from torch.utils.data import DataLoader
from model.wideResNet import WideResNet
from utils.metric import accuracy
from utils.interleave import interleave
from utils.weight_update import WeightEMA
from utils.tqdm import get_tqdm_config
from utils.loss_function import Loss_Semisupervised

from dataloader.CIFAR10_Loader import get_cifar10

from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm
from utils.Augmentation import RandomFlip, RandomPadandCrop, ToTensor

class MixMatchTrainer():
    def __init__(self, args):
        self.args = args

        root_dir = './result' # PROJECT directory
        self.experiment_dir = root_dir # 학습된 모델을 저장할 폴더 경로 정의 및 폴더 생성
        os.makedirs(self.experiment_dir, exist_ok=True)

        name_exp = "_".join([str(self.args.n_labeled), str(self.args.T)]) # 주요 하이퍼 파라미터로 폴더 저장 경로 지정 
        self.experiment_dir = os.path.join(self.experiment_dir, name_exp)
        os.makedirs(self.experiment_dir, exist_ok=True)

        # Data
        print("==> Preparing CIFAR10 dataset")
        transform_train = transforms.Compose([
            RandomPadandCrop(32),
            RandomFlip(),
            ToTensor()
        ]) # 학습에 사용할 data augmentation 정의

        transform_val = transforms.Compose([
            ToTensor()
        ]) # validation, test dataset에 대한 data augmentation 정의
           # 합성곱 신경망에 입력 될 수 있도록만 지정(Augmentation 사용하지 않는 것과 동일)

        train_labeled_set, train_unlabeled_set, val_set, test_set = \
            get_cifar10(
                data_dir=os.path.join(root_dir, 'data'),
                n_labeled=self.args.n_labeled,
                transform_train=transform_train,
                transform_val=transform_val
            ) # 앞에서 정의한 (def) get_cifar10 함수에서 train_labeled, train_unlabeled, validation, test dataset
        
        # DataLoader 정의
        self.labeled_loader = DataLoader(
            dataset=train_labeled_set,
            batch_size=self.args.batch_size,
            shuffle=True, num_workers=0, drop_last=True
        )

        self.unlabeled_loader = DataLoader(
            dataset=train_unlabeled_set,
            batch_size=self.args.batch_size,
            shuffle=True, num_workers=0, drop_last=True
        )

        self.val_loader = DataLoader(
            dataset=val_set, shuffle=False, num_workers=0, drop_last=False
        )

        self.test_loader = DataLoader(
            dataset=test_set, shuffle=False, num_workers=0, drop_last=False
        )

        # Build WideResNet
        print("==> Preparing WideResNet")
        self.model = self.create_model(ema=False)
        self.ema_model = self.create_model(ema=True)

        # Define loss functions
        self.criterion_train = Loss_Semisupervised()
        self.criterion_val = nn.CrossEntropyLoss().to(self.args.cuda)

        # Define optimizers
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr)
        self.ema_optimizer = WeightEMA(self.model, self.ema_model, lr=self.args.lr, alpha=self.args.ema_decay)

        # 학습 결과를 저장할 Tensorboard 정의
        self.writer = SummaryWriter(self.experiment_dir)

    def create_model(self, ema=False):
        # Build WideResNet & EMA model
        model = WideResNet(num_classes=10)
        model = model.to(self.args.cuda)

        if ema:
            for param in model.parameters():
                param.detach_()
            
        return model
    
    def train(self, epoch):
        # 모델 학습 함수
        losses_t, losses_x, losses_u, ws = 0.0, 0.0, 0.0, 0.0
        self.model.train()

        # iter & next remind
        # iter: list 내 batch size 만큼 랜덤하게 불러오게 하는 함수
        # next: iter 함수가 작동하도록 하는 명령어
        iter_labeled = iter(self.labeled_loader)
        iter_unlabeled = iter(self.unlabeled_loader)

        with tqdm(**get_tqdm_config(total=self.args.num_iter,
                leave=True, color='blue')) as pbar:
            for batch_idx in range(self.args.num_iter):
                # 코드 작성 후 iter&next가 정확히 작용하지 않는 경우가 있음을 확인
                # 다시 iter_labeled, iter_unlabeled를 정의해 학습에 문제가 없도록 다시 선언
                try:
                    inputs_x, targets_x = iter_labeled.next()
                except:
                    iter_labeled = iter(self.labeled_loader)
                    inputs_x, targets_x = iter_labeled.next()
                real_B = inputs_x.size(0)

                # Transform label to one-hot
                targets_x = torch.zeros(real_B, 10).scatter_(1, targets_x.view(-1,1).long(), 1)
                inputs_x, targets_x = inputs_x.to(self.args.cuda), targets_x.to(self.args.cuda)

                try:
                    tmp_inputs, _ = iter_unlabeled.next()
                except:
                    iter_unlabeled = iter(self.unlabeled_loader)
                    tmp_inputs, _ = iter_unlabeled.next()

                inputs_u1, inputs_u2 = tmp_inputs[0], tmp_inputs[1]
                inputs_u1, inputs_u2 = inputs_u1.to(self.args.cuda), inputs_u2.to(self.args.cuda)

                # Unlabeled data에 대한 실제 값 생성
                # 서로 다른 Augmentation 결과의 출력 값의 평균 계산
                # Temperature 값으로 실제 값 스케일링
                with torch.no_grad():
                    outputs_u1 = self.model(inputs_u1)
                    outputs_u2 = self.model(inputs_u2)

                    pt = (torch.softmax(outputs_u1, dim=1)+torch.softmax(outputs_u2, dim=1)) / 2
                    pt = pt**(1/self.args.T)

                    targets_u = pt / pt.sum(dim=1, keepdim=True)
                    targets_u = targets_u.detach()
                
                # MixUp
                # 서로 다른 이미지와 레이블을 섞는 작업
                # feature space 상에서 범주 별 Decision boundary를 정확하게 잡아주는 역할
                inputs = torch.cat([inputs_x, inputs_u1, inputs_u2], dim=0)
                targets = torch.cat([targets_x, targets_u, targets_u], dim=0)

                l_mixup = np.random.beta(self.args.alpha, self.args.alpha)
                l_mixup = max(l_mixup, 1-l_mixup)

                # inputs의 index를 섞어 서로 다른 범주끼리 섞도록 하는 역할
                B = inputs.size(0)
                random_idx = torch.randperm(B)

                inputs_a, inputs_b = inputs, inputs[random_idx]
                targets_a, targets_b = targets, targets[random_idx]

                mixed_input = l_mixup*inputs_a + (1-l_mixup)*inputs_b
                mixed_target = l_mixup*targets_a + (1-l_mixup)*targets_b

                # batch size 만큼 분할 진행 (2N, C, H, W) -> (N, C, H, W) & (N, C, H, W)
                # 앞 부분은 labeled, 뒷 부분은 unlabeled
                '''
                이렇게 하는 이유는 첫 B는 Label 데이터로 활용, 나중 B는 Unlabeled data로 활용하기 위함 (관용적 활용법)
                '''
                
                mixed_input = list(torch.split(mixed_input, real_B))
                mixed_input = interleave(mixed_input, real_B)

                logits = [self.model(mixed_input[0])] # for labeled
                for input in mixed_input[1:]:
                    logits.append(self.model(input)) # for unlabeled

                logits = interleave(logits, real_B) # interleave: 정확히 섞이었는지 확인
                logits_x = logits[0]
                logits_u = torch.cat(logits[1:], dim=0)

                loss_x, loss_u, w = \
                    self.criterion_train(self.args,
                                    logits_x, mixed_target[:real_B],
                                    logits_u, mixed_target[real_B:],
                                    epoch+batch_idx/self.args.num_iter) # Semi-supervised loss 계산

                loss = loss_x + w * loss_u

                # Backpropagation and Model parameter update
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.ema_optimizer.step()

                losses_x += loss_x.item()
                losses_u += loss_u.item()
                losses_t += loss.item()
                ws += w

                self.writer.add_scalars(
                    'Training steps', {
                        'Total_loss': losses_t/(batch_idx+1),
                        'Labeled_loss':losses_x/(batch_idx+1),
                        'Unlabeled_loss':losses_u/(batch_idx+1),
                        'W values': ws/(batch_idx+1)
                    }, global_step=epoch*self.args.batch_size+batch_idx
                )

                pbar.set_description(
                    '[Train(%4d/ %4d)-Total: %.3f|Labeled: %.3f|Unlabeled: %.3f]'%(
                        (batch_idx+1), self.args.num_iter,
                        losses_t/(batch_idx+1), losses_x/(batch_idx+1), losses_u/(batch_idx+1)
                    )
                )
                pbar.update(1)

            pbar.set_description(
                '[Train(%4d/ %4d)-Total: %.3f|Labeled: %.3f|Unlabeled: %.3f]'%(
                    epoch, self.args.epochs,
                    losses_t/(batch_idx+1), losses_x/(batch_idx+1), losses_u/(batch_idx+1)
                )
            )
        
        return losses_t/(batch_idx+1), losses_x/(batch_idx+1), losses_u/(batch_idx+1)

    @torch.no_grad()
    def validate(self, epoch, phase):
        self.ema_model.eval()

        # Train, Validation, Test dataset 에 대한 DataLoader를 정의
        if phase == 'Train':
            data_loader = self.labeled_loader
            c = 'blue'
        elif phase == 'Valid':
            data_loader = self.val_loader
            c = 'green'
        elif phase == 'Test ':        
            data_loader = self.test_loader
            c = 'red'

        losses = 0.0
        top1s, top5s = [], []

        with tqdm(**get_tqdm_config(total=len(data_loader),
                leave=True, color=c)) as pbar:
            for batch_idx, (inputs, targets) in enumerate(data_loader):
                inputs, targets = inputs.to(self.args.cuda), targets.to(self.args.cuda)
                targets = targets.type(torch.LongTensor).to(self.args.cuda)
        
                outputs = self.ema_model(inputs)
                loss = self.criterion_val(outputs, targets)
                # labeled dataset에 대해서만 손실함수 계산
                # torch.nn.CrossEntropyLoss()를 사용해서 손실함수 계산

                prec1, prec5 = accuracy(outputs, targets, topk=(1, 5))
                losses += loss.item()
                top1s.append(prec1)
                top5s.append(prec5)

                self.writer.add_scalars(
                    f'{phase} steps', {
                        'Total_loss': losses/(batch_idx+1),
                        'Top1 Acc': np.mean(top1s),
                        'Top5 Acc': np.mean(top5s)
                    }, global_step=epoch*self.args.batch_size+batch_idx
                )

                pbar.set_description(
                    '[%s-Loss: %.3f|Top1 Acc: %.3f|Top5 Acc: %.3f]'%(
                        phase,
                        losses/(batch_idx+1), np.mean(top1s), np.mean(top5s)
                    )
                )
                pbar.update(1)

            pbar.set_description(
                '[%s(%4d/ %4d)-Loss: %.3f|Top1 Acc: %.3f|Top5 Acc: %.3f]'%(
                    phase,
                    epoch, self.args.epochs,
                    losses/(batch_idx+1), np.mean(top1s), np.mean(top5s)
                )
            )

        return losses/(batch_idx+1), np.mean(top1s), np.mean(top5s)
```
___

## 2. FixMatch
![image](https://blog.kakaocdn.net/dn/DiVQc/btqO0fASFqk/BeatuIw8TOEq0fYxZNSRT0/img.png)

FixMatch는 2020년 NeurlPS에 혜성처럼 등장하여 매우 simple한 방법론을 가지고도 SOTA급의 performance를 보여줬던 방법론입니다. Semi-supervised Learning을 학습할 때 unlabeled data에 대해 대표적으로 두 가지 방법을 활용하는데요, 첫째로 artificial label을 이용한 pseudo-labeling과, 두번째로 consistency regularization입니다. FixMatch는 위 두 가지 방법을 동시에 진행하는데요, 세부적인 진행 절차는 아래와 같아요.

- #### 1. Unlabeled data에 대해 weak augmentation, strong augmentation을 적용합니다.
- #### 2. 각기 다른 augmentation이 적용된 두 데이터를 동일 Model에 forwarding 시킵니다.
- #### 3. Weak augmentation을 통과한 output에서 가장 probability가 높은 값의 class를 pseudo-labeling 해줍니다.
- #### 4. 마지막으로, 해당 pseudo-labeling이 적용된 데이터와 Strong augmentation이 적용된 데이터를 input으로 받은 모델의 output과 CrossEntropy 연산을 하여 loss를 계산하는 학습 방식을 거치게 됩니다.

MixMatch는 말로 간단하게 한다고 요약해보더라도 8단계 정도였는데, FixMatch는 단 4단계만에 설명이 끝나네요, 정말 간단하죠? 이제 아래의 코드를 통해 python 상에서는 어떻게 구현되는지 알아보겠습니다.
___

# Python Code
## 모델에 사용되는 하이퍼파라미터 정의 (argparse)
``` py
import argparse

# Argument 정의
def FixMatch_parser():
    parser = argparse.ArgumentParser(description="FixMatch from Scratch")
    
    # method arguments
    parser.add_argument('--n-labeled', type=int, default=500) # labeled dat의 수
    parser.add_argument('--n-classes', type=int, default=10) # Class의 수
    parser.add_argument("--expand-labels", action="store_true", 
                        help="expand labels to fit eval steps")

    # training hyperparameters
    parser.add_argument('--batch-size', type=int, default=64) # 배치 사이즈
    parser.add_argument('--total-steps', default=2**14, type=int) # iteration마다 Scheduler가 적용되기에, Epoch가 아닌, Total-step을 정의
    parser.add_argument('--eval-step', type=int, default=1024) # Evaluation Step의 수
    parser.add_argument('--lr', type=float, default=0.002) # Learning rate
    parser.add_argument('--weight-decay', type=float, default=5e-4) # Weight Decay 정도
    parser.add_argument('--nesterov', action='store_true', default=True) # Nesterov Momentum
    parser.add_argument('--warmup', type=float, default=0.0) # Warmup 정도

    parser.add_argument('--use-ema', action='store_true', default=True) # EMA 사용여부
    parser.add_argument('--ema-decay', type=float, default=0.999) # EMA에서 Decay 정도

    parser.add_argument('--mu', type=int, default=7) # Labeled data의 mu배를 Unlabeled 데이터의 개수로 정의하기 위한 함수 (근데 위 Trainer에서는 안 쓰임)
    parser.add_argument('--T', type=float, default=1.0) # Sharpening 함수에 들어가는 하이퍼 파라미터

    parser.add_argument('--threshold', type=float, default=0.95) # Pseudo-Labeling이 진행되는 Threshold 정의
    parser.add_argument('--lambda-u', type=float, default=1.0) # Loss 가중치 정도
    return parser
```
데이터 로더의 경우 MixMatch와 코드가 동일합니다. 왜냐하면 두 모델 모두 Labeled, Unlabeled 데이터 둘 다 활용하기 때문에 indexing을 통해 해당 데이터셋을 구축하는 코드는 동일하게 가져가도 문제되지 않기 때문입니다. 

아래는 Augmentation 코드입니다. MixMatch와 차이점이 있다면 조금 더 다양한 augmentation 기법이 존재하는데, 이는 weak, strong augmentation 두 경우를 고려해야하기 때문입니다. fixmatch_augment_pool 함수를 통해 모든 augmentation 기법을 list-up 합니다.
```py
import random
import numpy as np

import PIL
import PIL.ImageOps
import PIL.ImageEnhance
import PIL.ImageDraw
from PIL import Image

PARAMETER_MAX = 10

def _float_parameter(v, max_v):
    return float(v) * max_v / PARAMETER_MAX

def _int_parameter(v, max_v):
    return int(v * max_v / PARAMETER_MAX)

# Augmentation 함수들을 정의
def AutoContrast(img, **kwargs):
    return PIL.ImageOps.autocontrast(img)

def Brightness(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    return PIL.ImageEnhance.Brightness(img).enhance(v)

def Color(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    return PIL.ImageEnhance.Color(img).enhance(v)

def Contrast(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    return PIL.ImageEnhance.Contrast(img).enhance(v)

def CutoutAbs(img, v, **kwargs):
    w, h = img.size
    x0, y0 = np.random.uniform(0, w), np.random.uniform(0, h)
    x0, y0 = int(max(0, x0 - v / 2.)), int(max(0, y0 - v / 2.))

    x1, y1 = int(min(w, x0 + v)), int(min(h, y0 + v))

    xy = (x0, y0, x1, y1)
    # gray
    color = (127, 127, 127)
    img = img.copy()
    
    PIL.ImageDraw.Draw(img).rectangle(xy, color)
    return img

def Cutout(img, v, max_v, bias=0):
    if v == 0:
        return img
    v = _float_parameter(v, max_v) + bias
    v = int(v * min(img.size))
    return CutoutAbs(img, v)

def Equalize(img, **kwargs):
    return PIL.ImageOps.equalize(img)

def Identity(img, **kwargs):
    return img

def Invert(img, **kwargs):
    return PIL.ImageOps.invert(img)

def Posterize(img, v, max_v, bias=0):
    v = _int_parameter(v, max_v) + bias
    return PIL.ImageOps.posterize(img, v)

def Rotate(img, v, max_v, bias=0):
    v = _int_parameter(v, max_v) + bias
    if random.random() < 0.5:
        v = -v
    return img.rotate(v)

def Sharpness(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    return PIL.ImageEnhance.Sharpness(img).enhance(v)

def ShearX(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    if random.random() < 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, v, 0, 0, 1, 0))

def ShearY(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    if random.random() < 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, v, 1, 0))

def Solarize(img, v, max_v, bias=0):
    v = _int_parameter(v, max_v) + bias
    return PIL.ImageOps.solarize(img, 256 - v)

def SolarizeAdd(img, v, max_v, bias=0, threshold=128):
    v = _int_parameter(v, max_v) + bias
    if random.random() < 0.5:
        v = -v
    img_np = np.array(img).astype(np.int)
    img_np = img_np + v
    img_np = np.clip(img_np, 0, 255)
    img_np = img_np.astype(np.uint8)
    img = Image.fromarray(img_np)
    return PIL.ImageOps.solarize(img, threshold)

def TranslateX(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    if random.random() < 0.5:
        v = -v
    v = int(v * img.size[0])
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, v, 0, 1, 0))

def TranslateY(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    if random.random() < 0.5:
        v = -v
    v = int(v * img.size[1])
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, 0, 1, v))

# RandAugment를 사용하기 위한 전체 Augmentation List를 정의
def fixmatch_augment_pool():
    
    '''
    augs: 활용할 Augmentation의 전체집합
    '''
    
    augs = [(AutoContrast, None, None),
            (Brightness, 0.9, 0.05),
            (Color, 0.9, 0.05),
            (Contrast, 0.9, 0.05),
            (Equalize, None, None),
            (Identity, None, None),
            (Posterize, 4, 4),
            (Rotate, 30, 0),
            (Sharpness, 0.9, 0.05),
            (ShearX, 0.3, 0),
            (ShearY, 0.3, 0),
            (Solarize, 256, 0),
            (TranslateX, 0.3, 0),
            (TranslateY, 0.3, 0)]
            
    return augs
```
해당 Augmentation 중 Random하게 각각 weak, strong에 대한 augmentation을 호출하고, 아래의 코드를 통해 데이터에 각각 weak augmentation, strong augmentation을 적용시켜줍니다.
```py
import random
import numpy as np

from utils.Augmentation import CutoutAbs
from utils.Augmentation import fixmatch_augment_pool


# 위에서 구현된 Augmentpool에서 랜덤으로 선정하여 실제 Augmentation을 구현
class RandAugmentMC(object):
    
    def __init__(self, n, m):
        
        '''
        초기값 지정
        n: 1~
        m: 1~10
        augment_pool: augmentation 함수들이 모여있는 집합
        '''
        
        assert n >= 1
        assert 1 <= m <= 10
        
        self.n = n
        self.m = m
        self.augment_pool = fixmatch_augment_pool()
    
    def __call__(self, img):
        
        '''
        1. 함수가 불리면 augment_pool에서 n개만큼 선택
        2. m범위에서 랜덤하게 operation 강도를 선정
        3. 50$의 확률로 위 2가지 과정을 진행할지 결정
        4. 마지막에는 Cutout Augmentation 진행
        '''
        
        ops = random.choices(self.augment_pool, k=self.n)
        
        for op, max_v, bias in ops:
            v = np.random.randint(1, self.m)
            if random.random() < 0.5:
                img = op(img, v=v, max_v=max_v, bias=bias)

        img = CutoutAbs(img, int(32*0.5))
        
        return img
```
기타 코드는 MixMatch와 동일하고, 학습 코드는 차이가 있기 때문에 아래 학습 코드를 살펴보겠습니다.
```py
import os
import numpy as np

import torch
import torch.nn.functional as F

from model.wideResNet import WideResNet

from utils.metric import accuracy
from utils.scheduler import get_cosine_schedule_with_warmup
from utils.weight_update import WeightEMA
from utils.tqdm import get_tqdm_config

from dataloader.CIFAR10_Loader import get_cifar10

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm

# trainer를 정의
class FixMatchTrainer():
    
    # 초깃값 지정
    def __init__(self, args):
        
        # argment를 받아오기
        self.args = args
        
        # 각종 Directory를 지정
        root_dir = './result' ### Project Directory
        data_dir = os.path.join(root_dir, 'data') ### Data Directory
        
        self.experiment_dir = os.path.join(root_dir, 'results') ### 학습된 모델을 저장할 큰 폴더
        os.makedirs(self.experiment_dir, exist_ok=True)

        name_exp = "_".join([str(self.args.n_labeled), str(self.args.T)]) ### 학습된 모델을 저장할 세부 폴더 (하이퍼파라미터로 지정)
        self.experiment_dir = os.path.join(self.experiment_dir, name_exp)
        os.makedirs(self.experiment_dir, exist_ok=True)
        
        # Load Dataset (Labeled, Unlabeled, Valid, Test dataset)
        print("==> Preparing CIFAR10 dataset")
        labeled_set, unlabeled_set, val_set, test_set = get_cifar10(self.args, data_dir=data_dir)
        
        # DataLoader를 각각 정의 (Labeled, Unlabeled, Valid, Test dataset)                 
        self.labeled_loader = DataLoader(
            labeled_set,
            sampler=RandomSampler(labeled_set), ### RandomSampler: DataLoader(shuffle=True) 와 동일한 역할
            batch_size=self.args.batch_size,
            num_workers=0,
            drop_last=True
        )

        self.unlabeled_loader = DataLoader(
            unlabeled_set,
            sampler=RandomSampler(unlabeled_set),
            batch_size=self.args.batch_size,
            num_workers=0,
            drop_last=True
        )

        self.val_loader = DataLoader(
            val_set,
            sampler=SequentialSampler(val_set), ### SequentialSampler: DataLoader(shuffle=False) 와 동일한 역할
            batch_size=self.args.batch_size,
            num_workers=0,
            drop_last=True
        )

        self.test_loader = DataLoader(
            test_set,
            sampler=SequentialSampler(test_set),
            batch_size=self.args.batch_size,
            num_workers=0
        )

        # WideResNet모델 정의
        print("==> Preparing WideResNet")
        self.model = WideResNet(self.args.n_classes).to(self.args.cuda)
        
        # 모델의 Gradient 초기화 및 Loss Function을 정의
        self.model.zero_grad()
        self.criterion = torch.nn.CrossEntropyLoss().to(self.args.cuda)

        # Optimzer를 정의: params의 이름 내 bias, bn이 들어가지 않는 경우에만 weight_decay 적용
        no_decay = ['bias', 'bn']
        grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(
                nd in n for nd in no_decay)], 'weight_decay': self.args.weight_decay},
            {'params': [p for n, p in self.model.named_parameters() if any(
                nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ] 
        self.optimizer = torch.optim.SGD(grouped_parameters, lr=self.args.lr,
                            momentum=0.9, nesterov=self.args.nesterov)
        
        self.scheduler = get_cosine_schedule_with_warmup(self.optimizer,
                                                    self.args.warmup,
                                                    self.args.total_steps)
        
        # EMA Model을 쓸건지 안 쓸건지 명시
        if self.args.use_ema:  
            self.ema_model = WeightEMA(self.model, self.args.ema_decay)
        
        # Tensorboard에 기록할 객체 정의
        self.writer = SummaryWriter(self.experiment_dir)

        
    # train을 위한 함수
    def train(self, epoch):
        
        # total, labeled, unlabeled loss 초기화 및 Mask probs(Threshold를 넘었는지 여부를 표시한 것) 초기화
        losses_t, losses_x, losses_u, mask_probs = 0.0, 0.0, 0.0, 0.0
        
        # 훈련모드 전환
        self.model.train()
        
        # iter함수로 Labeled data 및 Unlabeled data 불러오기
        iter_labeled = iter(self.labeled_loader)
        iter_unlabeled = iter(self.unlabeled_loader)

        with tqdm(**get_tqdm_config(total=self.args.eval_step,
                leave=True, color='blue')) as pbar:
            
            for batch_idx in range(self.args.eval_step): ### eval_step: 1024 // batch_size: 64
                
                ### Labeled Data(각각 데이터와 Target)
                try:
                    inputs_x, targets_x = iter_labeled.next()
                except:
                    iter_labeled = iter(self.labeled_loader)
                    inputs_x, targets_x = iter_labeled.next()
                real_B = inputs_x.size(0)
                
                ### Unlabeled Data (각각 Weak Aug, Strong Aug)
                try:
                    (inputs_u_w, inputs_u_s), _ = iter_unlabeled.next()
                except:
                    iter_unlabeled = iter(self.unlabeled_loader)
                    (inputs_u_w, inputs_u_s), _ = iter_unlabeled.next()
                
                ### Labeled data, Weak_aug Unlabeled data, Strong_aug Unlabeled data Concat하여 Input으로 활용
                inputs = torch.cat((inputs_x, inputs_u_w, inputs_u_s), dim=0).to(self.args.cuda)
                targets_x = targets_x.type(torch.LongTensor)
                targets_x = targets_x.to(self.args.cuda)
                
                logits = self.model(inputs) ##### 예측값이 들어있음
                
                ### Labeled data와 Unlabeled data를 구분
                logits_x = logits[:real_B]
                logits_u_w, logits_u_s = logits[real_B:].chunk(2)
                del(logits)

                # Labeled data에 대한 loss계산
                loss_x = F.cross_entropy(logits_x, targets_x, reduction='mean')

                # Unlabeled data에 대한 loss계산
                pseudo_labels = torch.softmax(logits_u_w.detach()/self.args.T, dim=-1) 
                max_prob, targets_u = torch.max(pseudo_labels, dim=-1)
                mask = max_prob.ge(self.args.threshold).float() ##### mask: Threshold보다 크면 True, 작으면 False를 반환

                ### strong augmentation된 이미지에서 산출된 logit과 Pseudo label 사이 cross_entropy 계산
                loss_u = (F.cross_entropy(logits_u_s, targets_u, reduction='none')*mask).mean()

                ### Total loss: Labeled data loss와 Unlabeled data loss의 가중합
                loss = loss_x + self.args.lambda_u * loss_u
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
                if self.args.use_ema:
                    self.ema_model.step(self.model)
                
                self.model.zero_grad()
                
                ### Tensorboard를 위해 loss값들을 기록
                losses_x += loss_x.item()
                losses_u += loss_u.item()
                losses_t += loss.item()
                mask_probs += max_prob.mean().item()
                
                ### Print log
                self.writer.add_scalars(
                    'Training steps', {
                        'Total_loss': losses_t/(batch_idx+1),
                        'Labeled_loss':losses_x/(batch_idx+1),
                        'Unlabeled_loss':losses_u/(batch_idx+1),
                        'Mask probs': mask_probs/(batch_idx+1)
                    }, global_step=epoch*self.args.batch_size+batch_idx
                )

                pbar.set_description(
                    '[Train(%4d/ %4d)-Total: %.3f|Labeled: %.3f|Unlabeled: %.3f]'%(
                        (batch_idx+1), self.args.eval_step,
                        losses_t/(batch_idx+1), losses_x/(batch_idx+1), losses_u/(batch_idx+1)
                    )
                )
                pbar.update(1)

            pbar.set_description(
                '[Train(%4d/ %4d)-Total: %.3f|Labeled: %.3f|Unlabeled: %.3f]'%(
                    epoch, self.args.epochs,
                    losses_t/(batch_idx+1), losses_x/(batch_idx+1), losses_u/(batch_idx+1)
                )
            )
        return losses_t/(batch_idx+1), losses_x/(batch_idx+1), losses_u/(batch_idx+1)

    
    # Validation 함수 (MixMatch와 동일)
    @torch.no_grad()
    def validate(self, epoch, phase):
        if phase == 'Train': ### Train Loss
            data_loader = self.labeled_loader
            c = 'blue'
        elif phase == 'Valid': ### Valid Loss
            data_loader = self.val_loader
            c = 'green'
        elif phase == 'Test ': ### Test Loss
            data_loader = self.test_loader
            c = 'red'
        
        ### 값 초기화
        losses = 0.0
        top1s, top5s = [], []
        
        ### 데이터를 넣은 후 Output 및 Loss값, 정확도 산출
        with tqdm(**get_tqdm_config(total=len(data_loader),
                leave=True, color=c)) as pbar:
            for batch_idx, (inputs, targets) in enumerate(data_loader):
                inputs, targets = inputs.to(self.args.cuda), targets.to(self.args.cuda)
                targets = targets.type(torch.LongTensor).to(self.args.cuda)
                
                outputs = self.ema_model.ema(inputs)
                loss = self.criterion(outputs, targets)

                prec1, prec5 = accuracy(outputs, targets, topk=(1, 5))
                losses += loss.item()
                top1s.append(prec1)
                top5s.append(prec5)

                self.writer.add_scalars(
                    f'{phase} steps', {
                        'Total_loss': losses/(batch_idx+1),
                        'Top1 Acc': np.mean(top1s),
                        'Top5 Acc': np.mean(top5s)
                    }, global_step=epoch*self.args.batch_size+batch_idx
                )

                pbar.set_description(
                    '[%s-Loss: %.3f|Top1 Acc: %.3f|Top5 Acc: %.3f]'%(
                        phase,
                        losses/(batch_idx+1), np.mean(top1s), np.mean(top5s)
                    )
                )
                pbar.update(1)

            pbar.set_description(
                '[%s(%4d/ %4d)-Loss: %.3f|Top1 Acc: %.3f|Top5 Acc: %.3f]'%(
                    phase,
                    epoch, self.args.epochs,
                    losses/(batch_idx+1), np.mean(top1s), np.mean(top5s)
                )
            )

        return losses/(batch_idx+1), np.mean(top1s), np.mean(top5s)
```
물론 코드가 길긴 합니다만, MixMatch의 학습 코드에 비해서는 굉장히 많이 간소화되었습니다. 왜냐하면 모델이 그만큼 간단하기 때문이죠. weak-augmentation된 데이터에 대한 pseudo-labeling 정보를 바탕으로 Strong-augmented data와 CrossEntropy 연산을 통해 모델을 학습하게 됩니다.
___
## 3. FlexMatch
![image](https://www.microsoft.com/en-us/research/uploads/prod/2022/09/usb-1.png)

FlexMatch는 FixMatch의 단점을 보완한 모델이에요. FixMatch의 경우 해당 Class에 속할 확률값(SoftMax)이 일정 Threshold을 넘을 때 해당 Class에 속할 확률을 1로 pseudo-labeling하고, 만일 넘지 못한다면 그 샘플은 학습에 참여시키지 않습니다. 그렇기 때문에 제거되는 샘플이 제법 많이 늘어나게 되고, 자칫하면 샘플의 수가 충분하지 못한 상황이 생길 수 있는 단점이 있죠. 그래서 FlexMatch는 해당 문제를 해결하고자 Class 별로 '난이도'에 따라 서로 다른 Threshold를 적용하는 방법을 도입합니다. 자세한 사항은 코드를 통해 알아보겠습니다.

___
# Python Code
## 모델에 사용되는 하이퍼파라미터 정의 (argparse)

``` py
import argparse

# Argument 정의
def FlexMatch_parser():
    parser = argparse.ArgumentParser(description="FlexMatch from scratch")
    
    # method arguments
    parser.add_argument('--n-labeled', type=int, default=2000) # labeled dat의 수
    parser.add_argument('--n-classes', type=int, default=10) # Class의 수
    parser.add_argument("--expand-labels", action="store_true", 
                        help="expand labels to fit eval steps")

    # training hyperparameters
    parser.add_argument('--batch-size', type=int, default=64) # 배치 사이즈
    parser.add_argument('--total-steps', default=2**14, type=int) # iteration마다 Scheduler가 적용되기에, Epoch가 아닌, Total-step을 정의
    parser.add_argument('--eval-step', type=int, default=1024) # Evaluation Step의 수
    parser.add_argument('--lr', type=float, default=0.03) # Learning rate
    parser.add_argument('--weight-decay', type=float, default=5e-4) # Weight Decay 정도
    parser.add_argument('--nesterov', action='store_true', default=True) # Nesterov Momentum
    parser.add_argument('--warmup', type=float, default=0.0) # Warmup 정도

    parser.add_argument('--use-ema', action='store_true', default=True) # EMA 사용여부
    parser.add_argument('--ema-decay', type=float, default=0.999) # EMA에서 Decay 정도

    parser.add_argument('--mu', type=int, default=7) # Labeled data의 mu배를 Unlabeled 데이터의 개수로 정의하기 위한 함수 (근데 위 Trainer에서는 안 쓰임)
    parser.add_argument('--T', type=float, default=1.0) # Sharpening 함수에 들어가는 하이퍼 파라미터

    parser.add_argument('--threshold', type=float, default=0.95) # Pseudo-Labeling이 진행되는 Threshold 정의
    parser.add_argument('--lambda-u', type=float, default=1.0) # Loss 가중치 정도
    return parser
```
위에서 설명드린 대로, FlexMatch는 FixMatch와 매우 유사합니다. 다만, 학습 과정에서 pseudo-labeling의 과정에서 Class별 난이도를 부여한다는 점에서 차이점이 있는데요. 코드 또한 해당 부분만 차이가 있기 때문에, 해당 차이점이 나타나있는 코드 부분만 살펴보도록 하겠습니다.

```py
import os
import numpy as np

import torch
import torch.nn.functional as F

from model.wideResNet import WideResNet

from utils.metric import accuracy
from utils.scheduler import get_cosine_schedule_with_warmup
from utils.weight_update import WeightEMA
from utils.tqdm import get_tqdm_config

from dataloader.CIFAR10_Loader import get_cifar10

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm
# trainer를 정의
class FlexMatchTrainer():

    
    # 초깃값 지정
    def __init__(self, args, c_threshold):

        # argment를 받아오기
        self.args = args
        self.c_threshold = c_threshold
        
        # 각종 Directory를 지정
        root_dir = './result' ### Project Directory
        data_dir = os.path.join(root_dir, 'data') ### Data Directory
        
        self.experiment_dir = os.path.join(root_dir, 'results') ### 학습된 모델을 저장할 큰 폴더
        os.makedirs(self.experiment_dir, exist_ok=True)

        name_exp = "_".join([str(self.args.n_labeled), str(self.args.T)]) ### 학습된 모델을 저장할 세부 폴더 (하이퍼파라미터로 지정)
        self.experiment_dir = os.path.join(self.experiment_dir, name_exp)
        os.makedirs(self.experiment_dir, exist_ok=True)
        
        # Load Dataset (Labeled, Unlabeled, Valid, Test dataset)
        print("==> Preparing CIFAR10 dataset")
        labeled_set, unlabeled_set, val_set, test_set = get_cifar10(self.args, data_dir=data_dir)
        
        # DataLoader를 각각 정의 (Labeled, Unlabeled, Valid, Test dataset)                 
        self.labeled_loader = DataLoader(
            labeled_set,
            sampler=RandomSampler(labeled_set), ### RandomSampler: DataLoader(shuffle=True) 와 동일한 역할
            batch_size=self.args.batch_size,
            num_workers=0,
            drop_last=True
        )

        self.unlabeled_loader = DataLoader(
            unlabeled_set,
            sampler=RandomSampler(unlabeled_set),
            batch_size=self.args.batch_size,
            num_workers=0,
            drop_last=True
        )

        self.val_loader = DataLoader(
            val_set,
            sampler=SequentialSampler(val_set), ### SequentialSampler: DataLoader(shuffle=False) 와 동일한 역할
            batch_size=self.args.batch_size,
            num_workers=0,
            drop_last=True
        )

        self.test_loader = DataLoader(
            test_set,
            sampler=SequentialSampler(test_set),
            batch_size=self.args.batch_size,
            num_workers=0
        )

        # WideResNet모델 정의
        print("==> Preparing WideResNet")
        self.model = WideResNet(self.args.n_classes).to(self.args.cuda)
        
        # 모델의 Gradient 초기화 및 Loss Function을 정의
        self.model.zero_grad()
        self.criterion = torch.nn.CrossEntropyLoss().to(self.args.cuda)

        # Optimzer를 정의: params의 이름 내 bias, bn이 들어가지 않는 경우에만 weight_decay 적용
        no_decay = ['bias', 'bn']
        grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(
                nd in n for nd in no_decay)], 'weight_decay': self.args.weight_decay},
            {'params': [p for n, p in self.model.named_parameters() if any(
                nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ] 
        self.optimizer = torch.optim.SGD(grouped_parameters, lr=self.args.lr,
                            momentum=0.9, nesterov=self.args.nesterov)
        
        # Learning rate Scheduler를 적용

        self.scheduler = get_cosine_schedule_with_warmup(self.optimizer,
                                                    self.args.warmup,
                                                    self.args.total_steps)
        
        # EMA Model을 쓸건지 안 쓸건지 명시
        if self.args.use_ema:  
            self.ema_model = WeightEMA(self.model, self.args.ema_decay)
        
        # Tensorboard에 기록할 객체 정의
        self.writer = SummaryWriter(self.experiment_dir)

        
    # train을 위한 함수
    def train(self, epoch):
        
        # total, labeled, unlabeled loss 초기화 및 Mask probs(Threshold를 넘었는지 여부를 표시한 것) 초기화
        losses_t, losses_x, losses_u, mask_probs = 0.0, 0.0, 0.0, 0.0
        
        # 훈련모드 전환
        self.model.train()
        
        # iter함수로 Labeled data 및 Unlabeled data 불러오기
        iter_labeled = iter(self.labeled_loader)
        iter_unlabeled = iter(self.unlabeled_loader)
        
        over_threshold_count = [0] * 10 ### 각 Class별 초과한 Instance의 수를 초기화
        under_threshold_count = 0 ### 각 Class별 초과하지 않은 Instance 값을 초기화
        
        with tqdm(**get_tqdm_config(total=self.args.eval_step,
                leave=True, color='blue')) as pbar:
            
            for batch_idx in range(self.args.eval_step): ### eval_step: 1024 // batch_size: 64
                
                ### Labeled Data(각각 데이터와 Target)
                try:
                    inputs_x, targets_x = iter_labeled.next()
                except:
                    iter_labeled = iter(self.labeled_loader)
                    inputs_x, targets_x = iter_labeled.next()
                real_B = inputs_x.size(0)
                
                ### Unlabeled Data (각각 Weak Aug, Strong Aug)
                try:
                    (inputs_u_w, inputs_u_s), _ = iter_unlabeled.next()
                except:
                    iter_unlabeled = iter(self.unlabeled_loader)
                    (inputs_u_w, inputs_u_s), _ = iter_unlabeled.next()
                
                ### Labeled data, Weak_aug Unlabeled data, Strong_aug Unlabeled data Concat하여 Input으로 활용
                inputs = torch.cat((inputs_x, inputs_u_w, inputs_u_s), dim=0).to(self.args.cuda)
                targets_x = targets_x.type(torch.LongTensor)
                targets_x = targets_x.to(self.args.cuda)
                
                logits = self.model(inputs) ##### 예측값이 들어있음
                
                ### Labeled data와 Unlabeled data를 구분
                
                logits_x = logits[:real_B]
                logits_u_w, logits_u_s = logits[real_B:].chunk(2)
                del(logits)

                # Labeled data에 대한 loss계산
                loss_x = F.cross_entropy(logits_x, targets_x, reduction='mean')

                ### 예측 결과 --> return 예측한 Class, Threshold를 넘은 여부
                pseudo_labels = torch.softmax(logits_u_w.detach()/self.args.T, dim=-1) 
                max_prob, targets_u = torch.max(pseudo_labels, dim=-1)
                mask = torch.tensor([max_prob[idx].ge(self.c_threshold[idx]).float() for idx in targets_u])
                
                # Class를 넘은 여부를 기록
                for mask_value, class_idx in zip(mask, targets_u):
                    if mask_value == 0:
                        under_threshold_count += 1

                    elif mask_value == 1:
                        over_threshold_count[class_idx] += 1

                ### strong augmentation된 이미지에서 산출된 logit과 Pseudo label 사이 cross_entropy 계산
    
                logits_u_s = logits_u_s.to(self.args.cuda)
                targets_u = targets_u.to(self.args.cuda)
                mask = mask.to(self.args.cuda)
                loss_u = (F.cross_entropy(logits_u_s, targets_u, reduction='none')*mask).mean()
                
                ### Total loss: Labeled data loss와 Unlabeled data loss의 가중합
                loss = loss_x + self.args.lambda_u * loss_u
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
                if self.args.use_ema:
                    self.ema_model.step(self.model)
                
                self.model.zero_grad()
                
                ### Threshold를 Update 

                if max(over_threshold_count) < under_threshold_count: ### Warmup
                    for idx in range(10):
                        beta = over_threshold_count[idx] / max(max(over_threshold_count), under_threshold_count)
                        self.c_threshold[idx] = (beta/(2-beta)) * self.args.threshold

                else:
                    for idx in range(10):
                        beta = over_threshold_count[idx] / max(over_threshold_count) 
                        self.c_threshold[idx] = (beta/(2-beta)) * self.args.threshold
                
                ### Tensorboard를 위해 loss값들을 기록
                losses_x += loss_x.item()
                losses_u += loss_u.item()
                losses_t += loss.item()
                mask_probs += max_prob.mean().item()
                
                ### Print log
                self.writer.add_scalars(
                    'Training steps', {
                        'Total_loss': losses_t/(batch_idx+1),
                        'Labeled_loss':losses_x/(batch_idx+1),
                        'Unlabeled_loss':losses_u/(batch_idx+1),
                        'Mask probs': mask_probs/(batch_idx+1)
                    }, global_step=epoch*self.args.batch_size+batch_idx
                )

                pbar.set_description(
                    '[Train(%4d/ %4d)-Total: %.3f|Labeled: %.3f|Unlabeled: %.3f]'%(
                        (batch_idx+1), self.args.eval_step,
                        losses_t/(batch_idx+1), losses_x/(batch_idx+1), losses_u/(batch_idx+1)
                    )
                )
                pbar.update(1)

            pbar.set_description(
                '[Train(%4d/ %4d)-Total: %.3f|Labeled: %.3f|Unlabeled: %.3f]'%(
                    epoch, self.args.epochs,
                    losses_t/(batch_idx+1), losses_x/(batch_idx+1), losses_u/(batch_idx+1)
                )
            )
            
        print(f'Threshold_per_class: {self.c_threshold}')
        
        return losses_t/(batch_idx+1), losses_x/(batch_idx+1), losses_u/(batch_idx+1), self.c_threshold

    
    # Validation 함수 (MixMatch와 동일)
    @torch.no_grad()
    def validate(self, epoch, phase):
        if phase == 'Train': ### Train Loss
            data_loader = self.labeled_loader
            c = 'blue'
        elif phase == 'Valid': ### Valid Loss
            data_loader = self.val_loader
            c = 'green'
        elif phase == 'Test ': ### Test Loss
            data_loader = self.test_loader
            c = 'red'
        
        ### 값 초기화
        losses = 0.0
        top1s, top5s = [], []
        
        ### 데이터를 넣은 후 Output 및 Loss값, 정확도 산출
        with tqdm(**get_tqdm_config(total=len(data_loader),
                leave=True, color=c)) as pbar:
            for batch_idx, (inputs, targets) in enumerate(data_loader):
                
                targets = targets.type(torch.LongTensor)
                inputs, targets = inputs.to(self.args.cuda), targets.to(self.args.cuda)
                targets = targets.type(torch.LongTensor).to(self.args.cuda)
                
                outputs = self.ema_model.ema(inputs)
                loss = self.criterion(outputs, targets)

                prec1, prec5 = accuracy(outputs, targets, topk=(1, 5))
                losses += loss.item()
                top1s.append(prec1)
                top5s.append(prec5)

                self.writer.add_scalars(
                    f'{phase} steps', {
                        'Total_loss': losses/(batch_idx+1),
                        'Top1 Acc': np.mean(top1s),
                        'Top5 Acc': np.mean(top5s)
                    }, global_step=epoch*self.args.batch_size+batch_idx
                )

                pbar.set_description(
                    '[%s-Loss: %.3f|Top1 Acc: %.3f|Top5 Acc: %.3f]'%(
                        phase,
                        losses/(batch_idx+1), np.mean(top1s), np.mean(top5s)
                    )
                )
                pbar.update(1)

            pbar.set_description(
                '[%s(%4d/ %4d)-Loss: %.3f|Top1 Acc: %.3f|Top5 Acc: %.3f]'%(
                    phase,
                    epoch, self.args.epochs,
                    losses/(batch_idx+1), np.mean(top1s), np.mean(top5s)
                )
            )

        return losses/(batch_idx+1), np.mean(top1s), np.mean(top5s)
```
해당 코드는 FlexMatch의 학습 코드인데요, 가운데 쯤 보시게 되면 Class를 넘는 여부를 기록하고, Threshold를 Update하도록 설계되어 있습니다. 해당 학습 모델을 run하게 되면 매 epoch마다 Class별 threshold가 printing되게 설정되어 있습니다.
___
## Experimental Analysis

- Semi-supervised Learning을 검증하기 위해서는 우선 Supervised-Learning과 비교를 해야합니다. 단순 비교가 아니라, Labeled data의 비율을 줄여가며 실험 결과를 비교하는 것이 중요합니다. 왜냐하면 Semi-supervised Learning이 빛을 발할 때는 Labeled data가 많이 없을 때, 즉 학습 데이터의 비율이 낮을 때 Supervised Learning의 성능보다 높다면 의미가 있기 때문입니다. 따라서, 본 튜토리얼에서는 Cifar10 data의 labeled data의 수를 500, 2000, 10000, 15000, 20000 총 다섯 경우로 나누어 MixMatch, FixMatch, FlexMatch, 총 세 경우의 실험 결과를 비교하겠습니다. 

| Num_Label | Supervised(wideResNet)      |MixMatch          | FixMatch  | FlexMatch |
|:-------:|:--------------:|:-------------------:|:---------------:|:---------------:|
| 500     | 36.145%      | 54.108%     | 57.205% | **59.753%**           |
| 2,000     | 44.343%       | 80.570%           | 83.887%           | **84.027%**          |
| 10,000     | 69.173%     | 88.730%       | 89.570% | **89.580%**             |
| 15,000     | 75.991%        | 88.740%     | 90.550% | **90.695%**             |
| 20,000     | 82.255% | 89.220% |90.556%           |**90.934%**  |

실험 결과를 살펴보면, FlexMatch가 가장 성능이 좋았습니다. 사실 FixMatch와 그렇게 큰 성능 차가 나지는 않았습니다. 반면에 wideResNet을 사용한 지도 학습과 비교했을 때는 탁월한 성능 차이가 보였습니다. 아무래도 전체 Label 데이터 갯수가 60,000개인 Cifar-10 데이터셋에서 아주 적게(500, 2,000...)부터 많이는 약 30%(20,000)개만 썼기 때문에 Unlabeled data를 활용하는 준지도 학습의 경우보다 해당 경우에 있어서 만큼은 성능 차이가 돋보였던 것 같습니다. 아래는 해당 실험 결과를 보기 쉽게 plotting한 사진입니다.

![image](https://user-images.githubusercontent.com/115224653/209050586-8069c2a2-8df2-4dcc-a02b-43a2b08b7856.png)

plot으로 봤을 때 확실히 더 label data 수가 적을 때 준지도학습 방법론들의 성능이 좋았음을 알 수 있습니다. 대신 Label data의 수가 늘어남에 따라 지도 학습의 성능 상승 폭이 매우 커지는 것도 확인할 수 있습니다. 아마 머지 않아 준지도학습에 준하는, 오히려 그보다 좋은 성능을 보일 것처럼 상승하고 있습니다.


