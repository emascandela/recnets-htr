import torch
from torch import nn
import torch.nn.functional as F
from typing import Tuple, Dict, Any
from dataclasses import dataclass
from models.model_factory import ModelFactory
from models.base_model import BaseModel


@dataclass
class BaseParams:
    input_size: int
    num_filters: int
    block_size: int

    lstm_hidden_size: int
    lstm_layers: int
    num_outputs: int


@dataclass
class ShareParams:
    # share_layers: bool = False
    share_layer_stride: int = 1
    share_rnn_stride: int = 1
    share_layer_batch_norm: bool = False
    use_weight_scaler: bool = False


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
    ):
        super().__init__()
        self.use_weight_scaler = use_weight_scaler

        if use_last_conv:
            self.conv = BasicBlock._last_conv
        else:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel, padding="same")
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
