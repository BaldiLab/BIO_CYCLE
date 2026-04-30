import numpy as np
from torch import nn
import torch
from torch.multiprocessing import get_context
from collections import OrderedDict
from typing import List, Any
from argparse import ArgumentParser
import time
import os

from utils import data_utils, nnet_utils, table_utils, misc_utils
from utils.nnet_models import FullyConnectedModel, ResnetModel
from trajectory.trajectory import Trajectory
import pickle
import matplotlib.pyplot as plt
import hashlib
import re
from scipy.special import comb
from scipy.stats import combine_pvalues


class BioCycleBinaryNet(nn.Module):
    def _forward_unimplemented(self, *input_val: Any) -> None:
        pass

    def __init__(self, input_dim: int, act: str, hidden_dim: int):
        super().__init__()
        self.first_nn: nn.Module = nn.Linear(input_dim * 2, hidden_dim)
        self.resnet: nn.Module = ResnetModel(hidden_dim, 3, True, act)
        self.out_nn: nn.Module = FullyConnectedModel(hidden_dim, [1], [False], ["sigmoid"])

    def forward(self, x):
        x = self.first_nn(x)
        x = self.resnet(x)
        x = self.out_nn(x)

        return x


class BioCyclePeriodPhaseNet(nn.Module):
    def _forward_unimplemented(self, *input_val: Any) -> None:
        pass

    def __init__(self, input_dim: int, act: str, hidden_dim: int):
        super().__init__()
        self.first_nn: nn.Module = nn.Linear(input_dim * 2, hidden_dim)
        self.resnet: nn.Module = ResnetModel(hidden_dim, 3, True, act)
        self.lin_out: nn.Module = nn.Linear(hidden_dim, 3)

    def forward(self, x):
        x = self.first_nn(x)
        x = self.resnet(x)
        x = self.lin_out(x)

        periods = x[:, 0]

        sines_cosines = torch.tanh(x[:, 1:])
        sines = sines_cosines[:, 0] / torch.sqrt(torch.pow(sines_cosines[:, 0], 2) + torch.pow(sines_cosines[:, 1], 2))
        cosines = sines_cosines[:, 1] / torch.sqrt(
            torch.pow(sines_cosines[:, 0], 2) + torch.pow(sines_cosines[:, 1], 2))

        x = torch.stack((periods, sines, cosines), dim=1)

        return x


def get_null_cov_mat(tps) -> np.ndarray:
    tps = np.expand_dims(np.array(tps), 0)
    num_tps = tps.shape[1]

    eye = np.eye(num_tps)
    tps_repeat = np.repeat(tps, num_tps, axis=0)
    diff_mat_all = tps_repeat.transpose() - tps_repeat
    diff_mat_all2 = diff_mat_all ** 2

    length: float = 1.0
    noise: float = 0.01

    cov = np.exp(-diff_mat_all2 / (2 * (length ** 2))) + noise * eye

    return cov


def drop_inputs(data: np.ndarray) -> np.ndarray:
    num_tps: int = data.shape[1]
    max_drop: int = int(np.ceil(num_tps * 0.2))
    num_datapoints: int = data.shape[0]

    num_drop: np.array = np.random.randint(0, max_drop, size=num_datapoints)

    num_drop_tot: int = num_drop.sum()
    mask: np.ndarray = np.ones(data.shape)

    drop_idxs = np.random.choice(num_tps, size=num_drop_tot, replace=True)

    start_idx = 0
    for i, num_drop_i in enumerate(num_drop):
        end_idx: int = start_idx + num_drop_i

        drop_i = drop_idxs[start_idx:end_idx]

        mask[i, drop_i] = 0
        data[i, drop_i] = 0

        start_idx = end_idx

    data = np.concatenate((data, mask), axis=1)

    return data


def get_trajs_null(tps_targ: List[float]):
    print("at the beg of get trajs function:\t", len(tps_targ))
    # shift timepoints
    min_tp_targ = min(tps_targ)
    tps_targ = np.array([x - min_tp_targ for x in tps_targ])

    # get null trajs
    mean_trajs_file: str = "/home/kchangiz/codebase/BIO_CYCLE_2/null_dist/mean_traj.pkl"
    trajs_null_l: List[Trajectory] = pickle.load(open(mean_trajs_file, "rb"))
    print("shape of trajectory_l:\t", len(trajs_null_l))
    trajs_null: np.ndarray = np.stack([np.array(traj.mean) for traj in trajs_null_l], axis=0)

    tps_null: np.array = np.array(list(range(0, 48)))

    cov_null = get_null_cov_mat(tps_null)
    cov_null_inv = np.linalg.inv(cov_null)

    print("value of trajs_null and tps_targ:\t", len(trajs_null), len(tps_targ))

    trajs_mean = np.zeros((len(trajs_null), len(tps_targ)))

    for tp in tps_targ:
        pred_pos = np.where(tps_targ == tp)[0][0]

        if np.any(tps_null == tp):
            pred_pos_null = np.where(tps_null == tp)[0][0]
            trajs_mean[:, pred_pos] = trajs_null[:, pred_pos_null]
        else:
            tps_comb = np.sort(np.union1d(tps_null, np.array([tp])))
            cov_comb = get_null_cov_mat(tps_comb)
            pred_pos_cov = np.where(tps_comb == tp)[0][0]

            pred_cov_row = np.expand_dims(cov_comb[[pred_pos_cov], tps_comb != tp], 0)

            cov_mult = np.matmul(pred_cov_row, cov_null_inv)
            for traj_idx in range(trajs_null.shape[0]):
                trajs_mean[:, pred_pos] = np.matmul(cov_mult, trajs_null.T)

    # assert all([x in tps for x in tps_targ]), "Timepoints are not in those from trajectories from null dist"

    # trajs_mean: np.ndarray = np.stack([np.array(traj.mean) for traj in trajs], axis=0)
    # trajs_mean = trajs_mean[:, np.array(tps_targ).astype(np.int)]

    print("at the end of get trajs function:\t", trajs_mean.shape)
    return trajs_mean


def binary_data_gen_runner(num_itrs: int, batch_size: int, tps, start_period: float, end_period: float, data_queue):
    for _ in range(num_itrs):
        num_per_class: int = int(np.ceil(batch_size / 2))

        per_data = data_utils.get_data(tps, num_per_class, "periodic", period_range=(start_period, end_period))
        aper_data = data_utils.get_data(tps, num_per_class, "aperiodic")

        inputs = np.concatenate((per_data["waves"], aper_data["waves"]), axis=0)
        outputs = np.concatenate((np.ones(per_data["waves"].shape[0]), np.zeros(aper_data["waves"].shape[0])))
        outputs = np.expand_dims(outputs, 1)

        # normalize data
        inputs = data_utils.normalize_data(inputs)
        inputs = drop_inputs(inputs)

        # convert type
        inputs = inputs.astype(np.float32)
        outputs = outputs.astype(np.float32)

        data_queue.put((inputs, outputs))


def train_nnet_data_runner(nnet, nnet_save_dir: str, retrain: bool, data_runner_fn, tps: List[float],
                           start_period: float, end_period: float, lr: float, lr_d: float, batch_size: int,
                           num_itrs: int, device):
    misc_utils.makedirs_if_needed(nnet_save_dir)
    nnet_save_file: str = "%s/model_state_dict.pt" % nnet_save_dir

    if os.path.isfile(nnet_save_file) and (not retrain):
        print("Nnet trained and located at %s" % nnet_save_file)
        nnet = load_nnet(nnet_save_file, nnet, device)
        nnet.to(device)
    else:
        start_time = time.time()
        nnet.to(device)

        # start data runners
        ctx = get_context("spawn")
        num_procs: int = 10
        data_queue: ctx.Queue = ctx.Queue(num_procs)

        num_itrs_per_proc: List[int] = misc_utils.split_evenly(num_itrs, num_procs)
        procs: List = []
        for proc_id in range(num_procs):
            num_itrs_proc: int = num_itrs_per_proc[proc_id]
            proc = ctx.Process(target=data_runner_fn, args=(num_itrs_proc, batch_size, tps, start_period, end_period,
                                                            data_queue))
            proc.daemon = True
            proc.start()

            procs.append(proc)

        # train nnet
        nnet.train()
        nnet_utils.train_nnet(nnet, data_queue, num_itrs, "mse", lr, lr_d, device)

        nnet.eval()

        for proc in procs:
            proc.join()

        torch.save(nnet.state_dict(), nnet_save_file)

        print("Training time: %.2f seconds\n" % (time.time() - start_time))

    return nnet


def per_phase_data_gen_runner(num_itrs: int, batch_size: int, tps, start_period: float, end_period: float, data_queue):
    for _ in range(num_itrs):
        per_data = data_utils.get_data(tps, batch_size, "periodic", period_range=(start_period, end_period))

        inputs = per_data["waves"]

        periods = np.array(per_data["periods"])
        sines: np.array = np.sin(per_data["phases"])
        cosines: np.array = np.cos(per_data["phases"])

        outputs = np.stack((periods, sines, cosines), axis=1)

        # normalize data
        inputs = data_utils.normalize_data(inputs)
        inputs = drop_inputs(inputs)

        # convert type
        inputs = inputs.astype(np.float32)
        outputs = outputs.astype(np.float32)

        data_queue.put((inputs, outputs))


def calculate_phases(trajs: List[Trajectory], periods: np.array) -> np.array:
    phases: np.array = np.zeros(len(trajs))
    test_phases: np.array = np.arange(0, 2*np.pi, 0.1)
    for traj_idx, traj in enumerate(trajs):
        traj_mean: np.array = np.array(traj.mean_norm)
        tps: np.array = np.array(traj.tps)

        scores: List[float] = []
        for test_phase in test_phases:
            traj_mean_ref = np.cos((2.0 * np.pi / periods[traj_idx]) * tps - test_phase)
            score: float = float(np.sum(traj_mean * traj_mean_ref))
            scores.append(score)

        phases[traj_idx] = test_phases[np.argmax(scores)]

    return phases


def all_diff(arr: np.array) -> np.array:
    diffs_l: List[np.array] = []
    for i in range(len(arr) - 1):
        diffs_i = arr[i] - arr[i + 1:]
        diffs_l.append(diffs_i)

    return np.concatenate(diffs_l)


def calculate_amplitudes(trajs: List[Trajectory], periods: np.array, phases: np.array) -> np.array:
    amps: np.array = np.zeros(len(trajs))
    for traj_idx, traj in enumerate(trajs):
        traj_mean: np.array = np.array(traj.mean)

        tps: np.array = np.array(traj.tps)
        traj_mean_ref = np.cos((2.0 * np.pi / periods[traj_idx]) * tps - phases[traj_idx])

        diffs_traj = all_diff(traj_mean)
        diffs_traj_ref = all_diff(traj_mean_ref)

        eps = 1e-6 * np.ones(len(diffs_traj))
        amp = np.median(diffs_traj / (diffs_traj_ref + eps))
        if amp < 0:
            amp = np.median(np.abs(diffs_traj / (diffs_traj_ref + eps)))

        amps[traj_idx] = amp

    return amps


def calculate_offsets(trajs: List[Trajectory], periods: np.array, phases: np.array, amps: np.array) -> np.array:
    offsets: np.array = np.zeros(len(trajs))
    for traj_idx, traj in enumerate(trajs):
        period = periods[traj_idx]
        if period == 0:
            offset: float = 0
        else:
            traj_mean: np.array = np.array(traj.mean)

            tps: np.array = np.array(traj.tps)
            traj_mean_ref = amps[traj_idx] * np.cos((2.0 * np.pi / periods[traj_idx]) * tps - phases[traj_idx])

            offset: float = float(np.median(traj_mean - traj_mean_ref))

        offsets[traj_idx] = offset

    return offsets


def load_nnet(model_file: str, nnet: nn.Module, device: torch.device) -> nn.Module:
    # get state dict
    if device is None:
        state_dict = torch.load(model_file)
    else:
        state_dict = torch.load(model_file, map_location=device)

    # remove module prefix
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        k = re.sub('^module\.', '', k)
        new_state_dict[k] = v

    # set state dict
    nnet.load_state_dict(new_state_dict)

    nnet.eval()

    return nnet


def main():
    print("ITS HERE")
    # parse arguments
    parser: ArgumentParser = ArgumentParser()
    parser.add_argument('--input_file', type=str, required=True, help="")
    parser.add_argument('--output_dir', type=str, required=True, help="")

    parser.add_argument('--start_period', type=float, default=20.0, help="The lower bound of the range of periods to "
                                                                         "search for.")
    parser.add_argument('--end_period', type=float, default=28.0, help="The upper bound of the range of periods to "
                                                                       "search for")

    # nnet Training
    parser.add_argument('--num_hidden', type=int, default=1000, help="Number of hidden units")
    parser.add_argument('--act', type=str, default="splash", help="Type of activation function")
    parser.add_argument('--train_itrs', type=int, default=10 * 1000, help="Number of training iterations")
    parser.add_argument('--batch_size', type=int, default=100, help="Batch size for training")
    parser.add_argument('--nnet_dir', type=str, default="./saved_nnets", help="Directory for saving trained nnets")
    parser.add_argument('--retrain_binary', action='store_true', default=False,
                        help="Retrain binary nnet if already exists")
    parser.add_argument('--retrain_period', action='store_true', default=False,
                        help="Retrain nnet for estimating period if already exists")
    parser.add_argument('--lr', type=float, default=0.001, help="Initial learning rate")
    parser.add_argument('--pval_comb', type=str, default="friston", help="")
    parser.add_argument('--lr_d', type=float, default=0.99996, help="Learning rate decay for every iteration. "
                                                                    "Learning rate is decayed according to: "
                                                                    "lr * (lr_d ^ itr)")

    args = parser.parse_args()

    assert args.start_period <= args.end_period, "start_period must be less than or equal to end_period"

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)


    input_file_name = args.input_file.split("/")[-1]

    # parse file
    print("Parsing file")
    start_time = time.time()

    tps, _, num_reps, _ = table_utils.get_timepoints(args.input_file)
    num_timepoints: int = len(tps)

    print(tps, num_reps)

    if len(tps) < 2:
        out_file_name: str = args.output_dir + '/' + input_file_name
        with open(out_file_name, 'w') as error_file:
            error_file.write("Invalid input format: Please check the sample input and ensure the file and headers are formatted correctly.\n")
            return


    trajs: List[Trajectory] = table_utils.get_trajs(args.input_file)
    if trajs == None:
        input_file_name = args.input_file.split("/")[-1]
        out_file_name: str = args.output_dir + '/' + input_file_name
        with open(out_file_name, 'w') as error_file:
            error_file.write("Error:\n    A row with no values was found in your dataset.\n    Please check the dataset and run BioCycle again.")
        return

    print("%i datapoints with %i timepoints" % (len(trajs), num_timepoints))  # change ids to traj objects
    print("Timepoints(replicates): %s" % " ".join(["%s(%i)" % (tp, num_reps[tp]) for tp in tps]))

    print("Parse time: %.2f seconds\n" % (time.time() - start_time))

    # comput statistics
    print("Computing statistics")
    start_time = time.time()
    for traj in trajs:
        if traj.reps == []:
            input_file_name = args.input_file.split("/")[-1]
            out_file_name: str = args.output_dir + '/' + input_file_name
            with open(out_file_name, 'w') as error_file:
                error_file.write("Error:\n    A row with no values was found in your dataset.\n    Please check the dataset and run BioCycle again.")
            return
        traj.compute_statistics()

    print("Statistics computation time: %.2f seconds\n" % (time.time() - start_time))

    # normalize data
    print("Normalizing Data")
    start_time = time.time()
    for traj in trajs:
        traj.normalize()

    print("Normalizing data time: %.2f seconds\n" % (time.time() - start_time))

    # get nnet information
    device, devices, on_gpu = nnet_utils.get_device()
    print("device: %s, devices: %s, on_gpu: %s" % (device, devices, on_gpu))

    # train binary classificiation nnet
    tps_shifted_per_str: str = "_".join([str(x - min(tps)) for x in tps])
    tps_shifted_per_str = "%s_%s_%s" % (tps_shifted_per_str, args.start_period, args.end_period)
    tps_shifted_per_hash: str = hashlib.md5(tps_shifted_per_str.encode()).hexdigest()





    file_path = 'hashs.txt'  # Replace with your file path
    string_to_add = tps_shifted_per_hash

    with open(file_path, 'a') as file:
        file.write('\n' + tps_shifted_per_str + '\t' +  string_to_add)





    print("Training binary neural network")
    nnet_binary = BioCycleBinaryNet(num_timepoints, args.act, args.num_hidden)
    nnet_binary_save_dir: str = "%s/%s/nnet_binary/" % (args.nnet_dir, tps_shifted_per_hash)


    print(nnet_binary_save_dir)
    print(device)

    nnet_binary = train_nnet_data_runner(nnet_binary, nnet_binary_save_dir, args.retrain_binary, binary_data_gen_runner,
                                        tps, args.start_period, args.end_period, args.lr, args.lr_d, args.batch_size,
                                        args.train_itrs, device)

    # train period/phase nnet
    print("Training period network")
    nnet_per_phase = BioCyclePeriodPhaseNet(num_timepoints, args.act, args.num_hidden)
    nnet_per_phase_save_dir: str = "%s/%s/nnet_period/" % (args.nnet_dir, tps_shifted_per_hash)

    nnet_per_phase = train_nnet_data_runner(nnet_per_phase, nnet_per_phase_save_dir, args.retrain_period,
                                            per_phase_data_gen_runner, tps, args.start_period, args.end_period, args.lr,
                                            args.lr_d, args.batch_size, args.train_itrs, device)

    # normalize trajectory means
    traj_means: np.ndarray = np.zeros((len(trajs), num_timepoints))
    mask: np.ndarray = np.zeros((len(trajs), num_timepoints))
    for traj_idx, traj in enumerate(trajs):
        mask_i: np.array = np.array([tp in traj.tps for tp in tps])
        traj_means[traj_idx, mask_i] = traj.mean_norm
        mask[traj_idx] = mask_i

    traj_means_mask = np.concatenate((traj_means, mask), axis=1)
    traj_means_mask = traj_means_mask.astype(np.float32)

    # calculate outputs
    nnet_binary.eval()
    nnet_per_phase.eval()

    nnet_class_output = nnet_utils.forward_batched(nnet_binary, device, args.batch_size, traj_means_mask)[:, 0]
    nnet_perphase_output = nnet_utils.forward_batched(nnet_per_phase, device, args.batch_size, traj_means_mask)

    # calculate phase and lags
    periods = nnet_perphase_output[:, 0]
    periods = np.maximum(periods, args.start_period)
    periods = np.minimum(periods, args.end_period)

    print("Calculating phases")
    # sines = nnet_perphase_output[:, 1]
    # cosines = nnet_perphase_output[:, 2]
    # phases = misc_utils.sin_cos_to_phase(sines, cosines)
    phases = calculate_phases(trajs, periods)
    lags = phases * periods / (2.0 * np.pi)

    # amplitude
    print("Calculating amplitudes")
    amps = calculate_amplitudes(trajs, periods, phases)

    print("Calculating offsets")
    offsets = calculate_offsets(trajs, periods, phases, amps)

    print("Calculating p-values")
    # pvals mean periodicity
    trajs_null_mean: np.ndarray = get_trajs_null(tps)
    trajs_null_mean = data_utils.normalize_data(trajs_null_mean)

    mask: np.ndarray = np.ones(trajs_null_mean.shape)

    print("traj before shape", trajs_null_mean.shape)
    trajs_null_mean_mask: np.ndarray = np.concatenate((trajs_null_mean, mask), axis=1)
    trajs_null_mean_mask = trajs_null_mean_mask.astype(np.float32)

    print("trajs null mean size:\t", trajs_null_mean_mask.shape)   
    null_dist_mean = nnet_utils.forward_batched(nnet_binary, device, args.batch_size, trajs_null_mean_mask)[:, 0]
    print("nnet class output: \t", nnet_class_output)
    print("null dist mean:\t", null_dist_mean.max())
    print("size:\t", len(null_dist_mean))
    pvals_null_mean = np.array([np. mean(x < null_dist_mean) for x in nnet_class_output])
    print("pvals null mean:\t", pvals_null_mean)


    #plt.clf()
    #plt.hist(null_dist_mean)
    #plt.savefig("%s/null_dist_periodicity.jpg" % args.output_dir)
    #plt.close()

    #plt.clf()
    #plt.hist(pvals_null_mean)
    #plt.savefig("%s/pvals_periodicity.jpg" % args.output_dir)
    #plt.close()

    scatters = np.array([traj.scatter for traj in trajs])
    if max(list(num_reps.values())) > 1:
        num_null: int = 10000
        means = np.random.normal(0, 1, size=(num_null, len(tps)))
        reps_at_tp: List[np.ndarray] = [np.random.normal(0, 2.9, size=(num_null, num_reps[tp])) for tp in tps]

        trajs_null: List[Trajectory] = []
        for null_idx in range(num_null):
            reps: List[List[float]] = []
            for tp_idx, tp in enumerate(tps):

                reps_tp: List[float] = list(reps_at_tp[tp_idx][null_idx] + means[null_idx, tp_idx])
                reps.append(reps_tp)

            traj: Trajectory = Trajectory(reps, str(null_idx), tps, False, [[""]])
            trajs_null.append(traj)

        # pvals scatter
        for traj in trajs_null:
            traj.compute_statistics()

        null_dist_scatter: np.array = np.array([traj.scatter for traj in trajs_null])
        print("scatter min: \t", scatters)
        print("null_dist_scatter: \t", null_dist_scatter.min())
        pvals_null_scatter = np.array([np.mean(x > null_dist_scatter) for x in scatters])
        print("pvals null scatter:\t", pvals_null_scatter)

        plt.clf()
        plt.hist(null_dist_scatter)
        plt.savefig("%s/null_dist_scatter.jpg" % args.output_dir)
        plt.close()

        plt.clf()
        plt.hist(pvals_null_scatter)
        plt.savefig("%s/pvals_scatter.jpg" % args.output_dir)
        plt.close()

        # combine pvals
        if args.pval_comb == "friston":
            print("combine pvals: \t", pvals_null_mean, pvals_null_scatter)
            pvals = np.maximum(pvals_null_mean, pvals_null_scatter) ** 2
            print("pvals after evaluating:\t", pvals)
        elif args.pval_comb == "edgington":
            pvals_l = []
            for pval_mean, pval_scatter in zip(pvals_null_mean, pvals_null_scatter):
                sum_p = pval_mean + pval_scatter
                pval = (sum_p ** 2)/2.0
                if sum_p > 1:
                    pval = pval - comb(2, 1) * ((sum_p - 1)**2)/2

                pvals_l.append(pval)
            pvals = np.array(pvals_l)
        elif args.pval_comb == "fisher":
            pvals = np.array([combine_pvalues([x, y], method='fisher')[1]
                            for x, y in zip(pvals_null_mean, pvals_null_scatter)])
        elif args.pval_comb == "stouffer":
            pvals = np.array([combine_pvalues([x, y], method='stouffer')[1]
                            for x, y in zip(pvals_null_mean, pvals_null_scatter)])
        else:
            raise ValueError("Unknown p-value combination method %s" % args.pval_comb)

    else:
        pvals = pvals_null_mean

    print("final pvals:\t", pvals)
    qvals = misc_utils.bhq(pvals)

    # output results
    out_file_name: str = args.output_dir + '/' + input_file_name
    print("Writing output to %s" % out_file_name)
    start_time = time.time()

    print("TESTING TIME")
    _, _, _, time_list = table_utils.get_timepoints(args.input_file)
    _, input_all = table_utils.open_table(args.input_file)

    dataset_header = [f"TP_{int(time)}_REP_{i+1}" for time, reps in time_list.items() for i, _ in enumerate(reps)]

    header_out = ["ID", "P_VALUE", "Q_VALUE", "PERIOD", "LAG", "AMPLITUDE", "OFFSET", "MEAN_PERIODICITY", "SCATTER"]
    header_out = header_out + dataset_header
    #print(header_out)
    rows_out: List[List[str]] = []
    for idx in range(len(trajs)):
        
        row: List[str] = [trajs[idx].name, str(pvals[idx]), str(qvals[idx]), str(periods[idx]), str(lags[idx]),
                        str(amps[idx]), str(offsets[idx]), str(nnet_class_output[idx]), str(scatters[idx])]

        row_input: List[str] = []

        idx_list = []
        for cur_idx in time_list.values():
            idx_list.extend(cur_idx)


        if input_all[idx][0] != row[0]:
            row[0] = "THERE_IS_AN_ERROR" 

        for tp_rep_idx in idx_list:
            row_input.append(input_all[idx][tp_rep_idx])
        
        row = row + row_input

        rows_out.append(row)

    table_utils.write_table(out_file_name, header_out, rows_out)

    print("Write time: %.2f seconds\n" % (time.time() - start_time))

    print("Done")

if __name__ == "__main__":
    main()
