import csv
import re
from typing import Tuple, List, Set, Dict

import numpy as np
from utils import misc_utils
from trajectory.trajectory import Trajectory


label_names: List[str] = ["IS_PERIODIC", "PERIOD", "PHASE", "AMPLITUDE", "OFFSET", "P_VALUE", "Q_VALUE",
                          "MEAN_PERIODICITY", "SCATTER"]


def is_num_old(string: str):
    if string[0] == '-':
        string = string[1:]
    return all([i.isnumeric() for i in string.split('.', 1)])


def is_num(string: str):
    try:
        float(string)
        return True
    except ValueError:
        return False


def open_table(fname: str, delimiter: str = "\t") -> Tuple[List[str], List[List[str]]]:
    with open(fname) as file:
        reader = csv.reader(file, delimiter=delimiter)
        rows: List[List[str]] = [row for row in reader]
    header: List[str] = rows.pop(0)

    return header, rows


def get_header(fname: str, delimiter: str = "\t") -> List[str]:
    with open(fname) as file:
        reader = csv.reader(file, delimiter=delimiter)
        header = next(reader)

    return header


def get_timepoints(fname: str):
    header = get_header(fname)

    tp_set: Set[float] = set()
    tp_to_idxs: Dict[float, List[int]] = dict()
    for col_idx, col_name in enumerate(header):
        tp_regex = re.search("^([^0-9]+)?([0-9]+(\\.[0-9]+)?)", col_name)

        if tp_regex:
            tp: float = float(tp_regex.group(2))
            tp_set.add(tp)

            if tp not in tp_to_idxs:
                tp_to_idxs[tp] = []
            tp_to_idxs[tp].append(col_idx)

    tps: List[float] = list(np.sort([x for x in tp_set]))
    num_reps: Dict[float, int] = dict([(tp, len(tp_to_idxs[tp])) for tp in tps])

    tp_reps: List[float] = []
    for tp in tps:
        tp_reps.extend([tp] * num_reps[tp])

    return tps, tp_reps, num_reps, tp_to_idxs


def get_trajs(fname: str) -> List[Trajectory]:
    header, rows = open_table(fname)
    tps, _, _, tp_to_idxs = get_timepoints(fname)

    trajs: List[Trajectory] = []
    for row in rows:
        reps: List[List[float]] = []
        reps_raw: List[List[str]] = []
        tps_traj: List[float] = []
        missing_tps: bool = False
        for tp in tps:
            tp_idxs = tp_to_idxs[tp]
            
            try:
                reps_raw.append([row[tp_idx] for tp_idx in tp_idxs])
            except IndexError:
                return

            reps_tp: List[float] = [float(row[tp_idx]) for tp_idx in tp_idxs if is_num(row[tp_idx])]

            if len(reps_tp) > 0:
                reps.append(reps_tp)
                tps_traj.append(tp)
            else:
                missing_tps = True

        name: str = row[0]
        traj: Trajectory = Trajectory(reps, name, tps_traj, missing_tps, reps_raw)
        trajs.append(traj)

    return trajs


def get_labels(fname: str) -> Dict[str, np.array]:
    header, rows = open_table(fname)

    labels: Dict = dict()
    for label in label_names:
        if label not in header:
            continue

        label_vals = get_column(fname, label)
        if label == "IS_PERIODIC":
            if label_vals[0] in ['0', '1']:
                for idx in range(len(label_vals)):
                    if label_vals[idx] == "1":
                        label_vals[idx] = "TRUE"
                    elif label_vals[idx] == "0":
                        label_vals[idx] = "FALSE"
                    else:
                        raise ValueError("Unknown IS_PERIODIC label: %s" % label_vals[idx])

            labels[label] = [x.upper() == "TRUE" for x in label_vals]
        else:
            labels[label] = []
            for label_val in label_vals:
                try:
                    labels[label].append(float(label_val))
                except ValueError:
                    labels[label].append(np.nan)

    # convert to numpy format
    for label in labels.keys():
        labels[label] = np.array(labels[label])

    return labels


def add_labels(header: List[str], rows: List[List[str]], labels: Dict) -> Tuple[List[str], List[List[str]]]:
    for label in label_names:
        if label not in labels.keys():
            continue

        header.append(label.upper())
        for row, val in zip(rows, labels[label]):
            row.append(str(val))

    return header, rows


def trajs_to_rows(trajs: List[Trajectory]) -> List[List[str]]:
    rows: List[List[str]] = []
    for traj in trajs:
        reps_flat, _ = misc_utils.flatten(traj.reps_raw)
        row = [traj.name] + reps_flat
        rows.append(row)

    return rows


def get_column(fname: str, col_name: str) -> List[str]:
    header, rows = open_table(fname)

    col_idxs: int = header.index(col_name)
    vals: List[str] = [row[col_idxs] for row in rows]

    return vals


def write_table(fname: str, header: List[str], rows: List[List[str]], delimiter="\t"):
    with open(fname, "w") as file:
        writer = csv.writer(file, delimiter=delimiter)
        writer.writerow(header)

        for row in rows:
            writer.writerow(row)
