import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from typing import Tuple, Dict, Any, Literal, List
from dataclasses import dataclass
from functools import partial
from models.model_factory import ModelFactory
# from sklearn.cluster import KMeans
from fast_pytorch_kmeans import KMeans, init_methods

from models.base_model import BaseModel
from joblib import parallel_backend

def stack(data):
    if isinstance(data, torch.Tensor):
        return datas
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

class ClusterableModule(nn.Module):
    def get_clusterable_params(self):
        params = [] 
        for m in self.modules():
            if isinstance(m, ClusterableParam):
                params += m.get_clusterable_params()
        return params

    @staticmethod
    def __add_property(module, name, method):
        cls = type(module)
        if not hasattr(cls, '__perinstance'):
            cls_ = type(cls.__name__, (cls,), {})
            cls_.__perinstance = True
            module.__class__ = cls
        setattr(cls, name, property(method))

    def make_clusterable(self):
        for m in self.modules():
            if isinstance(m, ClusterableParam):
                m.make_clusterable()
    
    def set_clusterable_param(self, module_class, param, axis, avoid_params = None):
        if avoid_params is None:
            avoid_params = []

        c_param_name = "_clustered_"+param
        for idx, m in self.named_modules():
            if isinstance(m, module_class) and (idx not in avoid_params):
                original_param = getattr(m, param)
                clusterable_param = ClusterableParam(original_param, axis, group=module_class.__name__)
                setattr(m, c_param_name, clusterable_param)
                delattr(m, param)
                self.__add_property(m, param, lambda self: getattr(self, c_param_name).param)

    def cluster(self, cluster_ratio=1.0, replace_params=False):
        weight_list = self.get_clusterable_params()
        weights = {}
        for name, w in weight_list:
            if name not in weights:
                weights[name] = []
            weights[name].append(w)

        if cluster_ratio >= 1.0:
            return []
        
        print("clustering")

        with torch.no_grad():
            # weight_index = []

            shaped_groups = {}
            for k, weight_group in weights.items():
                for s in weight_group:
                    new_k = k+"_"+"_".join(map(str, s.param.shape))
                    if new_k in shaped_groups:
                        shaped_groups[new_k].append(s)
                    else:
                        shaped_groups[new_k] = [s]
            metrics = []

            for k, weight_group in shaped_groups.items():
                # n_clusters = cluster_ratio
                n_clusters = int(np.ceil(cluster_ratio * len(weight_group)))

                np_weights = np.stack([w.param.detach().cpu().numpy() for w in weight_group])
                weight_shape = np_weights.shape[1:]
                np_weights = np_weights.reshape([np_weights.shape[0], -1])
                # flatten_weight []
                # for w in weights["conv"]:
                #     # flatten_weights.append(w.detach().cpu().numpy().reshape([-1, np.prod(w.shape[-2:])]))
                #     # weight_index.extend(range(last_i, last_i+w))
                # flatten_weights = np.concatenate(flatten_weights)cpi

                # with parallel_backend("threading", n_jobs=16):
                    # kmeans = KMeans(n_clusters=n_clusters, random_state=0, init="random", n_init=1).fit(np_weights)
                print(k, n_clusters, weight_shape, np_weights.shape)
                kmeans = KMeans(n_clusters=n_clusters, init_method="random", minibatch=min(32, n_clusters))
                # kmeans = KMeans(n_clusters=n_clusters, init_method="kmeans++", minibatch=min(32, n_clusters))

                # print("Initialized")
                ws = torch.Tensor(np_weights).cuda()
                kmeans.fit(ws.half(), centroids=init_methods.init_methods[kmeans.init_method](ws, kmeans.n_clusters).half())
                print("fitted")
                labels = kmeans.predict(ws.half())

                metrics.append(
                    {
                        "name": k,
                        "n_clusters": n_clusters,
                        "labels": labels.cpu().numpy().tolist(),
                    }
                )

                # clustered_weights = np.zeros(len(np_weights)).astype(np.int32)
                print(labels, kmeans.centroids.shape)
                print("")
                clustered_weights = np.zeros_like(np_weights)

                # cluster_centers = torch.from_numpy(kmeans.cluster_centers_)
                cluster_centers = kmeans.centroids
                cluster_centers = cluster_centers.float()
                if replace_params:
                    cluster_centers = cluster_centers.reshape((-1, *weight_shape))
                    cluster_centers = cluster_centers.cuda()
                    cluster_centers = [nn.Parameter(c) for c in torch.unbind(cluster_centers)]

                    clustered_weights = [cluster_centers[c] for c in labels]

                    for w, new_w in zip(weight_group, clustered_weights):
                        w.param = new_w
                else:
                    for i, c in enumerate(labels-1):
                        clustered_weights[i] = cluster_centers[c].cpu().numpy()

                    for w, new_w in zip(weight_group, clustered_weights):
                        w.param.copy_(torch.Tensor(new_w.reshape(w.param.shape)))

            return metrics

class ClusterableParamWrapper(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.param = nn.Parameter(*args, **kwargs)

class ClusterableParam(ClusterableModule):
    def __init__(self, param, axis, group):
        super().__init__()
        self._p = param
        self.param_group = group
        self.is_clusterable = False
        self.shape = param.shape

        # assert len(param.shape) in (3, 4), "Error, unsuported number of dimensions"
        if isinstance(axis, int):
            axis = [axis]

        self.cluster_shape = [1]
        for ax, dim in enumerate(param.shape):
            if ax in axis:
                self.cluster_shape[0] *= dim
            else:
                self.cluster_shape.append(dim)

        if len(self.cluster_shape) == 1:
            self.cluster_shape.append(1)

        # if share_mode == "unit":
        #     self.cluster_shape = (np.prod(param.shape), 1)
        #     # self._weights = to_param(unbind(self.weight, range(len(self.weight.shape))))
        # elif share_mode == "channel":
        #     self.cluster_shape = (np.prod(param.shape[:2]), *param.shape[2:])
        #     # self._weights = to_param(unbind(self.weight, [0, 1]))
        # elif share_mode == "all" or share_mode is None:
        #     self.cluster_shape = (1, *(param.shape))
        #     # self._weights = self.weight
        # else:
        #     raise Exception("Unknown share mode", share_mode)


    
    @property
    def param(self):
        if self.is_clusterable:
            return torch.stack([p.param for p in self._cluster_p]).reshape(self.shape)
        
        return self._p

    def make_clusterable(self):
        if self.is_clusterable:
            return
        self._cluster_p = nn.ModuleList([ClusterableParamWrapper(p) for p in torch.unbind(self._p.reshape(self.cluster_shape))])
        del self._p
        self.is_clusterable = True
    
    def get_clusterable_params(self):
        if self.is_clusterable:
            return [(self.param_group, p) for p in self._cluster_p]
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
        # self._weights = nn.ModuleList([ClusterableParam(p) for p in torch.unbind(self.weight.reshape(self.cluster_shape))])
        # del self.weight
        # ClusterableConv2d.weight = property(lambda self: torch.stack([p.param for p in self._weights]).reshape(self.weight_shape))
        self._weight = ClusterableParam(self.weight, share_mode)
        ClusterableConv2d.weight = property(lambda self: self._weight.param)

        # self.bias/weight
    
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
    
    def get_clusterable_params(self):
        return self._weight.get_clusterable_params()
    
    def make_clusterable(self):
        self._weight.make_clusterable()

    # @property
    # def bias(self):
    #     return self._fuse_weights(self._weights)


@dataclass
class BaseParams:
    input_size: int
    num_filters: int
    const_filters: bool
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

    cnn_share_mode: Literal["unit", "channel", "all"] = "channel"
    lstm_share_mode: Literal["unit", "channel", "all"] = "channel"
    share_lstm: bool = False
    share_cnn: bool = False
    # share_mode: Literal["unit", "channel", "all"] = "channel"
    cluster_mode: Literal["per_layer", "all"] = "all"
    cluster_ratio: float = 0.25
    warmup_steps: int = 1
    cluster_steps: int = 1

    precision: Literal["FP", "Q8", "Q1.58"] = "FP"


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
        # share_mode: str = "all"
    ):
        super().__init__()
        self.use_weight_scaler = use_weight_scaler

        if use_last_conv:
            self.conv = BasicBlock._last_conv
        else:
            # self.conv = ClusterableConv2d(in_channels, out_channels, kernel, padding="same", share_mode=share_mode)
            self.conv = nn.Conv2d(in_channels, out_channels, kernel, padding=kernel//2)
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
    
    # def get_clusterable_params(self):
    #     return self.conv.get_clusterable_params()

    # def make_clusterable(self):
    #     self.conv.make_clusterable()
            


class CRNN(BaseModel, ClusterableModule):
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
        if self.base_params.const_filters:
            filters = [num_filters] * 5
        else:
            filters = [num_filters * (i+1) for i in range(5)]

        # self.conv = nn.Conv2d(3, num_filters, 1)

        self.block1 = self.make_block(3, filters[0], self.base_params.block_size, kernel_size=3, max_pool=(2, 2), dropout=0.0)
        self.block2 = self.make_block(filters[0], filters[1], self.base_params.block_size, kernel_size=3, max_pool=(2, 2), dropout=0.0)
        self.block3 = self.make_block(filters[1], filters[2], self.base_params.block_size, kernel_size=3, max_pool=(2, 2), dropout=0.2)
        self.block4 = self.make_block(filters[2], filters[3], self.base_params.block_size, kernel_size=3, max_pool=(1, 1), dropout=0.2)
        self.block5 = self.make_block(filters[3], filters[4], self.base_params.block_size, kernel_size=3, max_pool=(1, 1), dropout=0.2)

        self.cuda()

        lstm_layers = []
        lstm_input_shape = self.get_feature_shape()[-1]
        self.fc_prelstm = nn.Linear(lstm_input_shape, self.base_params.lstm_hidden_size*2, bias=False)

        lstm_input_shape = self.base_params.lstm_hidden_size*2
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
    
    # def get_clusterable_params(self):
    #     convs = {}
    #     recurrents = {}
    #     for block in [self.block1, self.block2, self.block3, self.block4, self.block5]:
    #         for layer in block:
    #             if isinstance(layer, BasicBlock):
    #                 w = layer.get_clusterable_params()
    #                 if self.share_params.cluster_mode == "per_layer":
    #                     convs[f"conv{len(convs)}"] = w
    #                 elif self.share_params.cluster_mode == "all":
    #                     if "conv" not in convs:
    #                         convs["conv"] = []
    #                     convs['conv'].extend(w)

    #     return {**convs, **recurrents}

    #n def make_clusterable(self):
    #n     for block in [self.block1, self.block2, self.block3, self.block4, self.block5]:
    #n         for layer in block:
    #n             if isinstance(layer, BasicBlock):
    #n                 layer.make_clusterable()

        # for layer in self.lstm_layers:
        #     recurrents.append(layer.weight_ih_l0)
        #     recurrents.append(layer.weight_hh_l0)
        #     recurrents.append(layer.weight_ih_l0_reverse)
        #     recurrents.append(layer.weight_hh_l0_reverse)
        
    
    def make_block(self, in_channels: int, out_channels: int, num_layers: int, kernel_size: int, max_pool: tuple[int, int], dropout: float) -> nn.Module:
        block = [
            nn.Conv2d(in_channels, out_channels, 1, bias=False)
        ]
        for i in range(0, num_layers, self.share_params.share_layer_stride):
            use_last_conv = (i % self.share_params.share_layer_stride) != 0

            block.append(BasicBlock(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    kernel=kernel_size,
                    use_last_conv=use_last_conv,
                    use_weight_scaler=self.share_params.use_weight_scaler,
                    dropout=dropout,
                    # share_mode=self.share_params.share_mode,
                )
            )

        block.append(nn.MaxPool2d(max_pool))
        return nn.Sequential(*block)
    
    def get_name(self) -> str:
        stride_suffix = f"-S{self.share_params.share_layer_stride}" if self.share_params.share_layer_stride != 1 else ""
        ws_suffix = "WS" if self.share_params.use_weight_scaler else ""
        lstm_suffix = f"-{self.base_params.lstm_layers}x{self.base_params.lstm_hidden_size}LSTM" + (f"-S{self.share_params.share_rnn_stride}" if self.share_params.share_rnn_stride != 1 else "")

        cluster_suffix = ""
        baseline_prefix = ""
        recursion_prefix = ""
        if self.share_params.cluster_ratio != 1.0:
            cluster_suffix += "_"
            cluster_suffix += f"r{self.share_params.cluster_ratio}"
            if self.share_params.share_cnn:
                cluster_suffix += f"_CNN{self.share_params.cnn_share_mode}"
            if self.share_params.share_lstm:
                cluster_suffix += f"_LSTM{self.share_params.lstm_share_mode}"
            cluster_suffix += f"_C{self.share_params.cluster_steps}"
            cluster_suffix += f"_W{self.share_params.warmup_steps}"
        elif (self.share_params.share_layer_stride > 1) or (self.share_params.share_rnn_stride > 1):
            recursion_prefix = "# "
        else:
            baseline_prefix = "* "
        
        precision_suffix = ""
        if self.share_params.precision != "FP":
            precision_suffix = f"_{self.share_params.precision}"
        
        return f"{baseline_prefix}{recursion_prefix}CRNN{self.base_params.block_size}x{self.base_params.num_filters}{'C' if self.base_params.const_filters else ''}{stride_suffix}{ws_suffix}{lstm_suffix}{cluster_suffix}{precision_suffix}"


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
        elif (share_params.cluster_ratio == 1.0) and (share_params.warmup_steps != 0 or share_params.cluster_steps != 0 or share_params.cnn_share_mode != "all" or share_params.lstm_share_mode != "all"):
            return False
        elif (not share_params.share_cnn) and (not share_params.share_cnn) and (share_params.warmup_steps != 0 or share_params.cluster_steps != 0 or share_params.cnn_share_mode != "all"):
            return False
        elif (not share_params.share_cnn) and (share_params.cnn_share_mode != "all"):
            return False
        elif (not share_params.share_lstm) and (share_params.lstm_share_mode != "all"):
            return False


        return True


ModelFactory.register(CRNN)
