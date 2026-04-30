from typing import Dict, List, Tuple
import numpy as np
from utils import misc_utils


def sine(tps: np.array, period: np.ndarray, phase: np.ndarray):
    return np.sin((2.0 * np.pi / period) * tps - phase)


def cosine(tps: np.ndarray, period: np.ndarray, phase: np.ndarray):
    return np.cos((2.0 * np.pi / period) * tps - phase)


def cosine2(tps: np.ndarray, period: np.ndarray, phase: np.ndarray):
    period2: np.ndarray = (1.0 / 3.0) * period

    phase1: np.ndarray = phase + 0.215*period/np.pi
    phase2: np.ndarray = (phase1 + 0.25 * period2) % period
    return cosine(tps, period, phase1)/1.39 + 0.5 * cosine(tps, period2, phase2)


def cosine_peak(tps: np.ndarray, period: np.ndarray, phase: np.ndarray):
    sign = 1
    return -1 + sign * 2 * (abs(cosine(tps, 2*period, phase/2)) ** 10)


def sine_triangle(tps: np.array, period: np.ndarray, phase: np.ndarray):
    phase = phase - np.pi / 2.0

    return (8 / (np.pi ** 2)) * (sine(tps, period, phase) - (1.0 / 9.0) * sine(tps, period/3.0, 3.0 * phase) +
                                 (1.0 / 25.0) * sine(tps, period/5.0, 5.0 * phase))


def sine_square(tps: np.array, period: np.ndarray, phase: np.ndarray):
    phase = phase - np.pi / 2.0

    return (4 / np.pi) * (sine(tps, period, phase) + (1.0 / 3.0) * sine(tps, period/3.0, 3.0 * phase) +
                          (1.0 / 5.0) * sine(tps, period / 5.0, 5.0 * phase))


def cosine_linear(tps: np.ndarray, period: np.ndarray, phase: np.ndarray, slope: np.ndarray):
    return cosine(tps, period, phase) + slope * tps


def cosine_plus_exp(tps: np.ndarray, period: np.ndarray, phase: np.ndarray, slope: np.ndarray):
    sign = np.sign(slope)
    slope = np.abs(slope)
    return cosine(tps, period, phase) + sign * np.exp(slope * tps / 2.0)


def cosine_times_exp(tps: np.ndarray, period: np.ndarray, phase: np.ndarray, slope: np.ndarray):
    return cosine(tps, period, phase) * np.exp(slope * tps)


def whitenoise(tps: np.ndarray, num: int):
    num_tps: int = tps.shape[1]
    return np.random.normal(0, 1, size=(num, num_tps))


def linear(tps: np.ndarray, slope: np.ndarray):
    return slope * tps


def flat(tps: np.ndarray, num: int):
    num_tps: int = tps.shape[1]
    signal = np.zeros((num, num_tps))

    return signal


def ar(tps: np.ndarray, num: int):
    # AR process of order one as mentioned in
    # http://www.ncbi.nlm.nih.gov/pmc/articles/PMC2881374/

    sigma = 1.0
    num_tps = tps.shape[1]
    white_noise = np.random.normal(0, sigma, size=(num, num_tps))

    n = (white_noise[:, 1:(num_tps - 1)] * white_noise[:, 2:num_tps]).sum(axis=1)
    d = (white_noise[:, 2:num_tps] ** 2.0).sum(axis=1)
    alpha = np.expand_dims(n / d, 1)

    signal = np.zeros((num, num_tps))
    signal[:, 0] = white_noise[:, 0]

    signal[:, 1:num_tps] = alpha * white_noise[:, 0:(num_tps - 1)] + np.random.normal(0, sigma, size=(num, num_tps-1))

    return signal


def gp_rbf(tps: np.ndarray, num: int):
    num_tps = tps.shape[1]
    signal = np.zeros((num, num_tps))

    gp_mean = np.zeros(num_tps)
    eye = np.eye(num_tps)
    tps_repeat = np.repeat(tps, num_tps, axis=0)
    diff_mat_all = tps_repeat.transpose() - tps_repeat
    diff_mat_all2 = diff_mat_all ** 2

    noises = np.random.uniform(0, 0.1, num)
    lengths = np.random.uniform(0.5, num_tps*2, num)
    noises = noises * np.random.randint(0, 2, num)

    for i in range(num):
        length = lengths[i]
        noise = noises[i]
        cov = np.exp(-diff_mat_all2 / (2 * (length ** 2))) + noise * eye
        signal_i = np.random.multivariate_normal(gp_mean, cov)

        signal[i] = signal_i

    return signal


def get_data(tps_l: List[float], num_gen: int, per_type: str, period_range: Tuple[float, float] = (20, 28),
             phase_range: Tuple[float, float] = (-2 * np.pi, 2 * np.pi), slope_range: Tuple[float, float] = (-.1, .1),
             amp_range: Tuple[float, float] = (1, 100), offset_range: Tuple[float, float] = (-100, 100),
             snr_range: Tuple[float, float] = (1, 8), outlier_prob: float = 0.05) -> Dict:
    # initialize
    num_tps: int = len(tps_l)
    tps: np.array = np.array(tps_l)
    tps = np.expand_dims(tps, 0)

    waves_l: List[np.ndarray] = []

    data: Dict = dict()
    data["periods"] = []
    data["phases"] = []
    data["wave_types"] = []

    # get wave types
    if per_type.upper() == "PERIODIC":
        # TODO fix cosine2
        wave_types = ["cosine", "peak", "triangle", "square", "cosine + linear",
                      "cosine + exp", "cosine * exp"]
    elif per_type.upper() == "APERIODIC":
        wave_types = ["whitenoise", "linear", "flat", "ar", "gp_rbf"]
    else:
        raise ValueError("Unknown per type %s" % per_type)

    wave_nums: List[int] = misc_utils.split_evenly(num_gen, len(wave_types))

    # get waves
    for wave_type, wave_num in zip(wave_types, wave_nums):
        if wave_num == 0:
            continue

        wave_type = wave_type.lower()

        # get parameters
        periods: np.ndarray = np.random.uniform(period_range[0], period_range[1], size=(wave_num, 1))
        phases: np.ndarray = np.random.uniform(phase_range[0], phase_range[1], size=(wave_num, 1))
        slopes: np.ndarray = np.random.uniform(slope_range[0], slope_range[1], size=(wave_num, 1))

        if wave_type == "cosine":
            waves_i = cosine(tps, periods, phases)
        elif wave_type == "cosine2":
            waves_i = cosine2(tps, periods, phases)
        elif wave_type == "peak":
            waves_i = cosine_peak(tps, periods, phases)
        elif wave_type == "triangle":
            waves_i = sine_triangle(tps, periods, phases)
        elif wave_type == "square":
            waves_i = sine_square(tps, periods, phases)
        elif wave_type == "cosine + linear":
            waves_i = cosine_linear(tps, periods, phases, slopes)
        elif wave_type == "cosine + exp":
            slopes = slopes * 0.7
            waves_i = cosine_plus_exp(tps, periods, phases, slopes)
        elif wave_type == "cosine * exp":
            slopes = slopes * 0.5
            waves_i = cosine_times_exp(tps, periods, phases, slopes)
        elif wave_type == "whitenoise":
            waves_i = whitenoise(tps, wave_num)
        elif wave_type == "linear":
            waves_i = linear(tps, slopes)
        elif wave_type == "flat":
            waves_i = flat(tps, wave_num)
        elif wave_type == "ar":
            waves_i = ar(tps, wave_num)
        elif wave_type == "gp_rbf":
            waves_i = gp_rbf(tps, wave_num)

        else:
            raise ValueError("Unknown wave type %s" % wave_type)

        if wave_type not in ["flat"]:
            # add noise to waves
            # make noise zero with prob
            noise_z_prob = 0.05
            noise = np.random.normal(0, 1, size=waves_i.shape)
            noise = noise * np.random.choice([0, 1], size=[noise.shape[0], 1], p=[noise_z_prob, 1-noise_z_prob])
            # if wave_type in ["linear"]:
            #    noise = noise * slopes

            snrs = np.random.uniform(snr_range[0], snr_range[1], size=waves_i.shape)
            waves_i = waves_i + noise / snrs

        # make outliers
        has_outliers = np.random.choice([True, False], size=waves_i.shape[0], p=[outlier_prob, 1 - outlier_prob])
        num_outliers_max: int = int(np.ceil(num_tps * 0.05))
        for i in range(waves_i.shape[0]):
            if has_outliers[i]:
                num_outliers = np.random.choice([1, num_outliers_max])
                outlier_idxs = np.random.choice(num_tps, size=num_outliers, replace=False)

                waves_i[i][outlier_idxs] = waves_i[i][outlier_idxs] + np.random.uniform(-20, 20, size=num_outliers)

        # change amplitude
        amplitudes = np.random.uniform(amp_range[0], amp_range[1], size=wave_num)
        waves_i = waves_i * np.expand_dims(amplitudes, 1)

        # add offset
        offsets = np.random.uniform(offset_range[0], offset_range[1], size=wave_num)
        waves_i = waves_i + np.expand_dims(offsets, 1)

        waves_l.append(waves_i)
        data["periods"].extend(list(periods[:, 0]))
        data["phases"].extend(list(phases[:, 0]))
        data["wave_types"].extend([wave_type] * wave_num)

    data["waves"] = np.concatenate(waves_l, axis=0)

    return data


def normalize_data(data: np.ndarray) -> np.ndarray:
    inputs_mean = np.expand_dims(data.mean(axis=1), 1)
    inputs_std = np.expand_dims(data.std(axis=1), 1)
    inputs_std[inputs_std == 0] = 1.0
    data = (data - inputs_mean) / inputs_std

    return data
