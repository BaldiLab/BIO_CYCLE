import numpy as np
from typing import List, Tuple

import torch
from torch import Tensor
from torch import nn
import torch.optim as optim
from torch.optim.optimizer import Optimizer

import time
import os


def make_batches(inputs: np.ndarray, outputs: np.ndarray,
                 batch_size: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    num_examples = outputs.shape[0]
    rand_idxs = np.random.choice(num_examples, num_examples, replace=False)
    inputs = inputs.astype(np.float32)
    outputs = outputs.astype(np.float32)

    start_idx = 0
    batches = []
    while (start_idx + batch_size) <= num_examples:
        end_idx = start_idx + batch_size

        idxs = rand_idxs[start_idx:end_idx]

        inputs_batch = inputs[idxs]
        outputs_batch = outputs[idxs]

        batches.append((inputs_batch, outputs_batch))

        start_idx = end_idx

    return batches


def get_device() -> Tuple[torch.device, List[int], bool]:
    device: torch.device = torch.device("cpu")
    devices: List[int] = []
    on_gpu: bool = False
    if ('CUDA_VISIBLE_DEVICES' in os.environ) and torch.cuda.is_available():
        device = torch.device("cuda:%i" % 0)
        devices = [int(x) for x in os.environ['CUDA_VISIBLE_DEVICES'].split(",")]
        on_gpu = True
    else:
        torch.set_num_threads(1)

    return device, devices, on_gpu


def forward_batched(nnet: nn.Module, device: torch.device, batch_size: int, data: np.ndarray) -> np.ndarray:
    nnet_output_l: List[np.ndarray] = []

    print("data shape:\t", data.shape)

    start_idx: int = 0
    while start_idx < data.shape[0]:
        end_idx = min(start_idx + batch_size, data.shape[0])
        data_i = data[start_idx:end_idx]

        nnet_output_i = nnet(torch.tensor(data_i, device=device)).cpu().data.numpy()

        nnet_output_l.append(nnet_output_i)

        start_idx = end_idx

    nnet_output = np.concatenate(nnet_output_l, axis=0)

    print("nnet output shape:\t", nnet_output.shape)
    return nnet_output


def train_nnet(nnet: nn.Module, data_queue, num_itrs: int, loss_type: str, lr: float,
               lr_d: float, device: torch.device, display: bool = True) -> float:
    # optimization
    display_itrs = 100
    if loss_type.upper() == "MSE":
        criterion = nn.MSELoss()
    elif loss_type.upper() == "BCE":
        criterion = nn.BCELoss()
    else:
        raise ValueError("Unknown loss %s" % loss_type)

    optimizer: Optimizer = optim.Adam(nnet.parameters(), lr=lr)

    # initialize status tracking
    start_time = time.time()

    # train network
    nnet.train()

    # batches: List[Tuple[np.ndarray, np.ndarray]] = make_batches(inputs, outputs, batch_size)

    last_loss: float = np.inf
    for train_itr in range(num_itrs):
        # zero the parameter gradients
        optimizer.zero_grad()
        lr_itr: float = lr * (lr_d ** train_itr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_itr

        # get data
        inputs_batch_np, targets_batch_np = data_queue.get()

        # send data to device
        targets_batch: Tensor = torch.tensor(targets_batch_np, device=device)
        inputs_batch: Tensor = torch.tensor(inputs_batch_np, device=device)

        # forward
        nnet_outputs_batch: Tensor = nnet(inputs_batch)

        # cost
        loss = criterion(nnet_outputs_batch, targets_batch)

        # backwards
        loss.backward()

        # step
        optimizer.step()

        last_loss = loss.item()
        # display progress
        if (train_itr % display_itrs == 0) and display:
            disp_str: str = "Itr: %i, lr: %.2E, loss: %.2f" % (train_itr, lr_itr, loss.item())

            if loss_type.upper() == "BCE":
                acc_tens: Tensor = (100.0 * (targets_batch == (nnet_outputs_batch > 0.5))).mean()
                acc: float = acc_tens.item()
                disp_str = "%s Acc: %.2f%%" % (disp_str, acc)

            disp_str = "%s Time: %.2f" % (disp_str, time.time() - start_time)
            print(disp_str)

            start_time = time.time()

    return last_loss
