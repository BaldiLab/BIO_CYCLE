from typing import List, Any
import numpy as np

import torch
from torch import Tensor
import torch.nn as nn
from torch.nn.parameter import Parameter


def get_act(act: str) -> nn.Module:
    act = act.upper()
    if act == "RELU":
        act_module = nn.ReLU()
    elif act == "SIGMOID":
        act_module = nn.Sigmoid()
    elif act == "SPLASH":
        act_module = SPLASH(3)
    elif act == "LINEAR":
        act_module = LinearUnit()
    else:
        raise ValueError("Un-defined activation type %s" % act)

    return act_module


class LinearUnit(nn.Module):
    def _forward_unimplemented(self, *input_val: Any) -> None:
        pass

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class SPLASH(nn.Module):
    def _forward_unimplemented(self, *input_val: Any) -> None:
        pass

    def __init__(self, num_hinges: int):
        super().__init__()
        assert num_hinges > 0, "Number of hinges should be greater than zero, but is %s" % num_hinges
        assert ((num_hinges + 1) % 2) == 0, "Number of hinges should be odd, but is %s" % num_hinges

        self.num_hinges: int = num_hinges
        self.num_each_side: int = int((self.num_hinges + 1)/2)

        self.hinges: List[float] = list(np.linspace(0, 2.5, self.num_each_side))

        self.coeffs_right: Parameter = Parameter(torch.cat((torch.ones(1), torch.zeros(self.num_each_side-1))),
                                                 requires_grad=True)
        self.coeffs_left: Parameter = Parameter(torch.zeros(self.num_each_side), requires_grad=True)

    def forward(self, x):
        output: Tensor = torch.zeros_like(x)

        # output for x > 0
        for idx in range(self.num_each_side):
            output = output + self.coeffs_right[idx] * torch.clamp(x - self.hinges[idx], min=0)

        # output for x < 0
        for idx in range(self.num_each_side):
            output = output + self.coeffs_left[idx] * torch.clamp(-x - self.hinges[idx], min=0)

        return output


class FullyConnectedModel(nn.Module):
    def _forward_unimplemented(self, *input_val: Any) -> None:
        pass

    def __init__(self, input_dim: int, layer_dims: List[int], layer_batch_norms: List[bool], layer_acts: List[str]):
        super().__init__()
        self.layers: nn.ModuleList[nn.ModuleList] = nn.ModuleList()

        # layers
        for layer_dim, batch_norm, act in zip(layer_dims, layer_batch_norms, layer_acts):
            module_list = nn.ModuleList()

            # linear
            module_list.append(nn.Linear(input_dim, layer_dim))

            # batch norm
            if batch_norm:
                module_list.append(nn.BatchNorm1d(layer_dim))

            # activation
            act_module = get_act(act)
            module_list.append(act_module)

            self.layers.append(module_list)

            input_dim = layer_dim

    def forward(self, x):
        x = x.float()

        module_list: nn.ModuleList
        for module_list in self.layers:
            for module in module_list:
                x = module(x)

        return x


class ResnetModel(nn.Module):
    def _forward_unimplemented(self, *input_val: Any) -> None:
        pass

    def __init__(self, resnet_dim: int, num_resnet_blocks: int, batch_norm: bool, act: str):
        super().__init__()
        self.blocks = nn.ModuleList()

        # resnet blocks
        for block_num in range(num_resnet_blocks):
            block_net = FullyConnectedModel(resnet_dim, [resnet_dim] * 2, [batch_norm] * 2, [act, "LINEAR"])
            act_module = get_act(act)

            module_list: nn.ModuleList = nn.ModuleList([block_net, act_module])
            self.blocks.append(module_list)

    def forward(self, x):
        # resnet blocks
        module_list: nn.ModuleList
        for block in self.blocks:
            res_inp = x
            x = block[0](x)
            x = block[1](x + res_inp)

        return x
