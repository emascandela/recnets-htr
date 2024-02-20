import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from typing import Tuple, Dict, Any, Literal
from dataclasses import dataclass
from models.model_factory import ModelFactory
from models.base_model import BaseModel

def stack(data):
    if isinstance(data, torch.Tensor):
        return data
    return torch.stack([stack(d) for d in data])

def unbind(data, dims=[], depth=0):
    if not isinstance(dims, list):
        dims = list(dims)
    if len(dims) == 0:
        return data

    dims = sorted(dims)

    data = torch.unbind(data, dims[0]-depth)
    return [unbind(d, dims[1:], depth=depth+1) for d in data]
    
def flatten(data):
    if isinstance(data, list):
        if isinstance(data[0], list):
            return [di for d in data for di in flatten(d)]
    return data

def to_param(data):
    if isinstance(data, list):
        return [to_param(d) for d in data]
    return nn.Parameter(data)

def reshape(data, shape):
    if len(shape) == 0:
        return data

    assert shape[-1] != 0
    assert len(data) % shape[-1] == 0

    shape[-1]
    dim_data = []
    new_data = []
    for i, d in enumerate(data):
        dim_data.append(d)
        if i % (shape[-1]-1) == 0:
            new_data.append(dim_data)
            dim_data = []
    return reshape(new_data, shape[:-1])

class ClusterableParamWrapper(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.param = nn.Parameter(*args, **kwargs)

class ClusterableParam(nn.Module):
    def __init__(self, param, cluster_shape):
        super().__init__()
        self._p = param
        self.cluster_shape = cluster_shape
        self.is_clusterable = False
    
    @property
    def param(self):
        if self.is_clusterable:
            torch.stack([p.param for p in self._p])
        
        return self._p

    def make_clusterable(self):
        param = self._p
        self._p = nn.ModuleList([ClusterableParamWrapper(p) for p in torch.unbind(param.reshape(self.cluster_shape))])
        del param
        self.is_clusterable = True
    
    def get_clusterable_weights(self):
        if self.is_clusterable:
            return self._p
        return []


class ClusterableConv2d(nn.Conv2d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size,
        stride=1,
        padding=0,
        # dilation: _size_2_t = 1,
        # groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros',  # TODO: refine this type
        device=None,
        dtype=None,
        share_mode=None
    ):
        super().__init__(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias, padding_mode=padding_mode, device=device, dtype=dtype)
        self.weight_shape = self.weight.shape
        if share_mode == "unit":
            self.cluster_shape = (np.prod(self.weight.shape), 1)
            # self._weights = to_param(unbind(self.weight, range(len(self.weight.shape))))
        elif share_mode == "channel":
            self.cluster_shape = (np.prod(self.weight.shape[:2]), *self.weight.shape[2:])
            # self._weights = to_param(unbind(self.weight, [0, 1]))
        elif share_mode == "all" or share_mode is None:
            self.cluster_shape = (1, *(self.weight.shape))
            # self._weights = self.weight
        else:
            raise Exception("Unknown share mode", share_mode)

        # self._weights = nn.ModuleList([ClusterableParam(p) for p in torch.unbind(self.weight.reshape(self.cluster_shape))])
        # del self.weight
        # ClusterableConv2d.weight = property(lambda self: torch.stack([p.param for p in self._weights]).reshape(self.weight_shape))
        self._weight = ClusterableParam(self.weight, self.cluster_shape)
        ClusterableConv2d.weight = property(lambda self: self._weight.param)

        # self.bias
    
    # def device(self, *args, **kwargs):
    #     super().device(*args, **kwargs)
    #     for w in self._weights:
    #         w.device(*args, **kwargs)
    
    # def parameters(self, *args, **kwargs):
    #     for p in super().parameters(*args, **kwargs):
    #         yield p
    #     for p in self._weights:
    #         yield p.param
    
    # def set_weights(self, weights, replace=False):
    #     for i, w in enumerate(weights):
    #         if replace:
    #             self._weights[i] = w
    #         else:
    #             self._weights[i].copy_(w)
    
    def get_clusterable_weights(self):
        return self._weight.get_clusterable_weights()
    
    def make_clusterable(self):
        self._weight.make_clusterable()

    # @property
    # def bias(self):
    #     return self._fuse_weights(self._weights)


@dataclass
class BaseParams:
    input_size: int
    num_filters: int
    block_size: int

    lstm_hidden_size: int
    lstm_layers: int
    num_outputs: int

# class WCConv2d(nn.Conv2d):

@dataclass
class ShareParams:
    # share_layers: bool = False
    share_layer_stride: int = 1
    share_rnn_stride: int = 1
    share_layer_batch_norm: bool = False
    use_weight_scaler: bool = False
    share_mode: Literal["unit", "channel", "all"] = "channel"
    # share_mode: Literal["unit", "channel", "all"] = "channel"
    cluster_mode: Literal["per_layer", "all"] = "all"
    cluster_ratio: float = 0.25
    warmup_steps: int = 1
    cluster_steps: int = 1


class BasicBlock(nn.Module):
    _last_conv = None
    _last_bn = None

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel: int,
        use_last_conv: bool = False,
        use_last_bn: bool = False,
        use_weight_scaler: bool = False,
        dropout: float = 0.0,
        share_mode: str = "all"
    ):
        super().__init__()
        self.use_weight_scaler = use_weight_scaler

        if use_last_conv:
            self.conv = BasicBlock._last_conv
        else:
            self.conv = ClusterableConv2d(in_channels, out_channels, kernel, padding="same", share_mode=share_mode)
            BasicBlock._last_conv = self.conv
        
        self.dropout = nn.Dropout(dropout)
        
        if use_last_bn:
            self.bn = BasicBlock._last_bn
        else:
            self.bn = nn.BatchNorm2d(out_channels)
            BasicBlock._last_bn = self.bn

        if self.use_weight_scaler:
            self.ws = nn.Parameter(
                torch.tensor(1.0, dtype=torch.float32),
                requires_grad=True,
            )
    
    def forward(self, x):
        x = self.dropout(x)
        x = self.conv(x)
        x = self.bn(x)
        x = F.leaky_relu(x, 0.2)
        return x
    
    def get_clusterable_weights(self):
        return self.conv.get_clusterable_weights()

    def make_clusterable(self):
        self.conv.make_clusterable()
            


class CRNN(BaseModel):
    name: str = "crnn"

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

        num_filters = self.base_params.num_filters

        # self.conv = nn.Conv2d(3, num_filters, 1)

        self.block1 = self.make_block(3, num_filters, self.base_params.block_size, kernel_size=3, max_pool=(2, 2), dropout=0.0)
        self.block2 = self.make_block(num_filters, num_filters * 2, self.base_params.block_size, kernel_size=3, max_pool=(2, 2), dropout=0.0)
        self.block3 = self.make_block(num_filters * 2, num_filters * 3, self.base_params.block_size, kernel_size=3, max_pool=(2, 2), dropout=0.2)
        self.block4 = self.make_block(num_filters * 3, num_filters * 4, self.base_params.block_size, kernel_size=3, max_pool=(1, 1), dropout=0.2)
        self.block5 = self.make_block(num_filters * 4, num_filters * 5, self.base_params.block_size, kernel_size=3, max_pool=(1, 1), dropout=0.2)

        self.cuda()

        lstm_layers = []
        lstm_input_shape = self.get_feature_shape()[-1]
        self.fc_prelstm = nn.Linear(lstm_input_shape, self.base_params.lstm_hidden_size*2, bias=False)

        lstm_input_shape = self.base_params.lstm_hidden_size*2
        print(lstm_input_shape)
        for i in range(self.base_params.lstm_layers):
            if (i % self.share_params.share_rnn_stride != 0):
                l = lstm_layers[-1]
            else:
                l = nn.LSTM(
                    input_size=lstm_input_shape,
                    hidden_size=self.base_params.lstm_hidden_size,
                    num_layers=1,
                    bidirectional=True,
                    batch_first=True
                )
            # lstm_input_shape = self.base_params.lstm_hidden_size*2
            lstm_layers.append(l)
        self.lstm_layers = nn.ModuleList(lstm_layers)

        self.dropout = nn.Dropout(0.5)
        self.fc_out = nn.Linear(self.base_params.lstm_hidden_size*2, self.base_params.num_outputs)
    
    def get_clusterable_weights(self):
        convs = {}
        recurrents = {}
        for block in [self.block1, self.block2, self.block3, self.block4, self.block5]:
            for layer in block:
                if isinstance(layer, BasicBlock):
                    w = layer.get_clusterable_weights()
                    if self.share_params.cluster_mode == "per_layer":
                        convs[f"conv{len(convs)}"] = w
                    elif self.share_params.cluster_mode == "all":
                        if "conv" not in convs:
                            convs["conv"] = []
                        convs['conv'].extend(w)

    def make_clusterable(self):
        for block in [self.block1, self.block2, self.block3, self.block4, self.block5]:
            for layer in block:
                if isinstance(layer, BasicBlock):
                    layer.make_clusterable()

        # for layer in self.lstm_layers:
        #     recurrents.append(layer.weight_ih_l0)
        #     recurrents.append(layer.weight_hh_l0)
        #     recurrents.append(layer.weight_ih_l0_reverse)
        #     recurrents.append(layer.weight_hh_l0_reverse)
        
        return {**convs, **recurrents}
    
    def make_block(self, in_channels: int, out_channels: int, num_layers: int, kernel_size: int, max_pool: tuple[int, int], dropout: float) -> nn.Module:
        block = [
            nn.Conv2d(in_channels, out_channels, 1, bias=False)
        ]
        print(in_channels, out_channels)
        for i in range(0, num_layers, self.share_params.share_layer_stride):
            use_last_conv = (i % self.share_params.share_layer_stride) != 0
            print(use_last_conv)

            block.append(BasicBlock(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    kernel=kernel_size,
                    use_last_conv=use_last_conv,
                    use_weight_scaler=self.share_params.use_weight_scaler,
                    dropout=dropout,
                    share_mode=self.share_params.share_mode,
                )
            )

        block.append(nn.MaxPool2d(max_pool))
        return nn.Sequential(*block)
    
    def get_name(self) -> str:
        stride_suffix = f"-S{self.share_params.share_layer_stride}" if self.share_params.share_layer_stride != 1 else ""
        ws_suffix = "WS" if self.share_params.use_weight_scaler else ""
        lstm_suffix = f"-{self.base_params.lstm_layers}x{self.base_params.lstm_hidden_size}LSTM" + (f"-S{self.share_params.share_rnn_stride}" if self.share_params.share_rnn_stride != 1 else "")
        return f"CRNN{self.base_params.block_size}x{self.base_params.num_filters}{stride_suffix}{ws_suffix}{lstm_suffix}"


    def extract_features(self, x):
        # x = self.conv(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)

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
        # x = F.relu(self.fc(x))
        x = self.dropout(x)
        x = self.fc_prelstm(x)
        for lstm in self.lstm_layers:
            x, _ = lstm(x)
        x = self.dropout(x)
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
        elif base_params.block_size < share_params.share_layer_stride:
            return False

        return True


ModelFactory.register(CRNN)
