"""
Properly implemented ResNet-s for CIFAR10 as described in paper [1].

The implementation and structure of this file is hugely influenced by [2]
which is implemented for ImageNet and doesn't have option A for identity.
Moreover, most of the implementations on the web is copy-paste from
torchvision's resnet and has wrong number of params.

Proper ResNet-s for CIFAR10 (for fair comparision and etc.) has following
number of layers and parameters:

name      | layers | params
ResNet20  |    20  | 0.27M
ResNet32  |    32  | 0.46M
ResNet44  |    44  | 0.66M
ResNet56  |    56  | 0.85M
ResNet110 |   110  |  1.7M
ResNet1202|  1202  | 19.4m

which this implementation indeed has.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
[2] https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py

If you use this implementation in you work, please don't forget to mention the
author, Yerlan Idelbayev.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from torch.autograd import Variable

__all__ = [
    "ResNet",
    "resnet20",
    "resnet32",
    "resnet44",
    "resnet56",
    "resnet110",
    "resnet1202",
]


def _weights_init(m):
    classname = m.__class__.__name__
    # print(classname)
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1
    _last_conv1 = None
    _last_conv2 = None
    _last_bn1 = None
    _last_bn2 = None
    _last_shortcut = None

    def __init__(
        self,
        in_planes,
        planes,
        stride=1,
        option="A",
        use_last_conv: bool = False,
        use_last_bn: bool = False,
        use_weight_scaler: bool = False,
    ):
        super(BasicBlock, self).__init__()
        self.use_weight_scaler = use_weight_scaler

        if isinstance(stride, int):
            stride = (stride, stride)

        if use_last_conv:
            self.conv1 = BasicBlock._last_conv1
            self.conv2 = BasicBlock._last_conv2
        else:
            self.conv1 = nn.Conv2d(
                in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
            )
            self.conv2 = nn.Conv2d(
                planes, planes, kernel_size=3, stride=1, padding=1, bias=False
            )
            BasicBlock._last_conv1 = self.conv1
            BasicBlock._last_conv2 = self.conv2

        if use_last_bn:
            self.bn1 = self._last_bn1
            self.bn2 = self._last_bn2
        else:
            self.bn1 = nn.BatchNorm2d(planes)
            self.bn2 = nn.BatchNorm2d(planes)
            BasicBlock._last_bn1 = self.bn1
            BasicBlock._last_bn2 = self.bn2
        if self.use_weight_scaler:
            self.ws1 = nn.Parameter(
                torch.tensor(1.0, dtype=torch.float32),
                requires_grad=True,
            )
            self.ws2 = nn.Parameter(
                torch.tensor(1.0, dtype=torch.float32),
                requires_grad=True,
            )

        self.shortcut = nn.Sequential()
        if stride != (1, 1) or in_planes != planes:
            if option == "A":
                """
                For CIFAR10 ResNet paper uses option A.
                """
                if in_planes == planes:
                    self.shortcut = LambdaLayer(
                        lambda x: x[:, :, ::stride[0], ::stride[1]],
                    )
                else:
                    self.shortcut = LambdaLayer(
                        lambda x: F.pad(
                            x[:, :, ::stride[0], ::stride[1]],
                            (0, 0, 0, 0, planes // 4, planes // 4),
                            "constant",
                            0,
                        )
                    )

            elif option == "B":
                self.shortcut = nn.Sequential(
                    nn.Conv2d(
                        in_planes,
                        self.expansion * planes,
                        kernel_size=1,
                        stride=stride,
                        bias=False,
                    ),
                    nn.BatchNorm2d(self.expansion * planes),
                )

    def forward(self, x):
        out = x
        if self.use_weight_scaler:
            out = out * self.ws1
        out = F.relu(self.bn1(self.conv1(out)))
        if self.use_weight_scaler:
            out = out * self.ws2
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(
        self,
        block,
        num_blocks,
        num_classes=10,
        share_layer_stride: int = 1,
        share_layer_batch_norm: bool = False,
        use_weight_scaler: bool = False,
        first_layer_stride: int = 1,
    ):
        super(ResNet, self).__init__()
        self.in_planes = 16
        self.share_layer_stride = share_layer_stride
        self.share_layer_batch_norm = share_layer_batch_norm
        self.use_weight_scaler = use_weight_scaler

        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=first_layer_stride, padding=2, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=(2, 2))
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=(2, 1))
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=(2, 1))
        self.linear = nn.Linear(64, num_classes)

        # self.apply(_weights_init)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for i, stride in enumerate(strides):
            use_last_conv = (i != 0) and ((i - 1) % self.share_layer_stride != 0)
            use_last_bn = use_last_conv and self.share_layer_batch_norm
            # print(i, use_last_conv)
            # st = 2

            # 0  0  0
            # 1  0  0
            # 2  1  1
            # 3  0  1
            # 4  1  0
            # 5  0  1
            # 6  1  1

            if i < 2:
                layers.append(block(self.in_planes, planes, stride, use_weight_scaler=self.use_weight_scaler))
            else:
                layers.append(
                    block(
                        self.in_planes,
                        planes,
                        stride,
                        use_last_conv=use_last_conv,
                        use_last_bn=use_last_bn,
                        use_weight_scaler=self.use_weight_scaler,
                    )
                )
            # print(self.in_planes, planes, stride)
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def resnet20(*args, **kwargs):
    return ResNet(BasicBlock, [3, 3, 3], *args, **kwargs)


def resnet32(*args, **kwargs):
    return ResNet(BasicBlock, [5, 5, 5], *args, **kwargs)


def resnet44(*args, **kwargs):
    return ResNet(BasicBlock, [7, 7, 7], *args, **kwargs)


def resnet56(*args, **kwargs):
    return ResNet(BasicBlock, [9, 9, 9], *args, **kwargs)


def resnet110(*args, **kwargs):
    return ResNet(BasicBlock, [18, 18, 18], *args, **kwargs)


def resnet1202(*args, **kwargs):
    return ResNet(BasicBlock, [200, 200, 200], *args, **kwargs)


from typing import Tuple, Dict, Any
from dataclasses import dataclass
from models.model_factory import ModelFactory
from models.base_model import BaseModel


@dataclass
class BaseParams:
    input_size: int
    lstm_hidden_size: int
    lstm_layers: int
    num_outputs: int
    first_layer_stride: int
    variant: str


@dataclass
class ShareParams:
    # share_layers: bool = False
    share_layer_stride: int = 1
    share_rnn_stride: int = 1
    share_layer_batch_norm: bool = False
    use_weight_scaler: bool = False


class CFResNet(BaseModel):
    name: str = "cf_resnet"

    def __init__(
        self,
        base_params: Dict[str, Any],
        share_params: Dict[str, Any],
        train_params: Dict[str, Any],
        dataset_name: str,
    ):
        super().__init__()
        self.base_params = BaseParams(**base_params)
        self.share_params = ShareParams(**share_params)
        self.train_params = train_params
        self.dataset_name = dataset_name

        if self.base_params.variant == "r20":
            model_fn = resnet20
        elif self.base_params.variant == "r32":
            model_fn = resnet32
        elif self.base_params.variant == "r44":
            model_fn = resnet44
        elif self.base_params.variant == "r56":
            model_fn = resnet56
        elif self.base_params.variant == "r110":
            model_fn = resnet110

        self.model = model_fn(
            # share_layers=self.share_params.share_layers,
            first_layer_stride=self.base_params.first_layer_stride,
            share_layer_stride=self.share_params.share_layer_stride,
            share_layer_batch_norm=self.share_params.share_layer_batch_norm,
            num_classes=0,
            use_weight_scaler=self.share_params.use_weight_scaler,
        )

        self.cuda()

        lstm_layers = []
        for i in range(self.base_params.lstm_layers):
            if (i % self.share_params.share_rnn_stride):
                l = lstm_layers[-1]
            else:
                l = nn.LSTM(
                    input_size=self.base_params.lstm_hidden_size*2,
                    hidden_size=self.base_params.lstm_hidden_size,
                    num_layers=1,
                    bidirectional=True,
                    batch_first=True
                )
            lstm_layers.append(l)
        self.lstm_layers = nn.ModuleList(lstm_layers)

        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(self.get_feature_shape()[-1], self.base_params.lstm_hidden_size*2)
        self.fc_out = nn.Linear(self.base_params.lstm_hidden_size*2, self.base_params.num_outputs)
    
    def get_name(self) -> str:
        stride_suffix = f"-S{self.share_params.share_layer_stride}" if self.share_params.share_layer_stride != 1 else ""
        ws_suffix = "WS" if self.share_params.use_weight_scaler else ""
        lstm_suffix = f"-{self.base_params.lstm_layers}LSTM" + (f"-S{self.share_params.share_rnn_stride}" if self.share_params.share_rnn_stride != 1 else "")
        return f"ResNet{self.base_params.variant[1:]}{stride_suffix}{ws_suffix}{lstm_suffix}"


    def extract_features(self, x):
        x = F.relu(self.model.bn1(self.model.conv1(x)))
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)

        x = x.permute(0, 3, 1, 2)
        x = torch.flatten(x, start_dim=2)
        return x
   
    def get_feature_shape(self):
        inp = torch.zeros((1, 3, self.base_params.input_size, 128))
        inp = inp.cuda()
        # print(inp)
        out = self.extract_features(inp)
        return out.shape[1:]

    def forward(self, x):
        x = self.extract_features(x)
        x = F.relu(self.fc(x))
        x = self.dropout(x)
        for lstm in self.lstm_layers:
            x, _ = lstm(x)
        x = self.fc_out(x)
        return x


    @staticmethod
    def is_legal(base_params: Dict[str, Any], share_params: Dict[str, Any]) -> bool:
        base_params = BaseParams(**base_params)
        share_params = ShareParams(**share_params)

        if share_params.share_layer_batch_norm and share_params.share_layer_stride == 1:
            return False
        elif share_params.use_weight_scaler and share_params.share_layer_stride == 1:
            return False
        elif base_params.lstm_layers < share_params.share_rnn_stride:
            return False
        elif base_params.variant == "r20" and share_params.share_layer_stride > 2:
            return False
        elif base_params.variant == "r32" and share_params.share_layer_stride > 4:
            return False
        elif base_params.variant == "r44" and share_params.share_layer_stride > 6:
            return False
        elif base_params.variant == "r56" and share_params.share_layer_stride > 8:
            return False
        elif base_params.variant == "r110" and share_params.share_layer_stride > 17:
            return False

        return True


ModelFactory.register(CFResNet)
