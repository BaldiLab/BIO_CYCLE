from typing import List, Optional
import numpy as np


class Trajectory:
    def __init__(self, reps: List[List[float]], name: str, tps: List[float], missing_tps: bool,
                 reps_raw: List[List[str]]):
        self.reps_raw = reps_raw
        self.reps: List[List[float]] = reps
        self.mean: List[Optional[float]] = []
        self.mean_norm: List[Optional[float]] = []
        self.scatter: Optional[float] = None

        self.tps: List[float] = tps
        self.missing_tps: bool = missing_tps

        self.name = name

    def compute_statistics(self):
        max_num_reps: int = max([len(x) for x in self.reps])

        if max_num_reps == 1:
            self.mean: List[float] = [x[0] for x in self.reps]
            reps_std_mean: float = 0.0
        else:
            self.mean: List[float] = []
            reps_std: List[float] = []
            for vals_tp in self.reps:
                self.mean.append(float(np.mean(vals_tp)))
                reps_std.append(float(np.std(vals_tp)))

            reps_std_mean: float = float(np.mean(reps_std))

        mean_std: float = float(np.std(self.mean))
        if (mean_std == 0) and (reps_std_mean == 0):
            self.scatter = 0.0
        elif (mean_std == 0) and (reps_std_mean > 0):
            self.scatter = 6.0
        else:
            self.scatter = reps_std_mean/mean_std

    def normalize(self):
        mean = np.mean(self.mean)
        std = np.std(self.mean)
        if std == 0:
            std = 1.0

        self.mean_norm = (self.mean - mean) / std
