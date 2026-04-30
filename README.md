# BioCycle

BioCycle is a deep learning-based method for detecting rhythmicity in biological time-series data. It uses neural networks to classify whether a gene or feature exhibits periodic behavior, and estimates the period, phase, amplitude, and offset of the oscillation.

---

## Requirements

Install all dependencies with:

```bash
pip install -r requirements.txt
```

Dependencies: `numpy`, `torch`, `scipy`, `matplotlib`, `pandas`

---

## Setup

Before running, add the current directory to your `PYTHONPATH`:

```bash
source setup.sh
```

---

## Usage

```bash
python BioCycle.py --input_file <input_file.tsv> --output_dir <output_directory/>
```

### Input Format

A tab-separated file (`.tsv`) where the first row is a header with timepoint labels and each subsequent row is a gene/feature with its expression values across timepoints.

### Output

A tab-separated results file written to `--output_dir` containing the following columns:

| Column | Description |
|---|---|
| `ID` | Gene / feature name |
| `P_VALUE` | P-value for rhythmicity |
| `Q_VALUE` | BH-adjusted q-value |
| `PERIOD` | Estimated period (in same units as timepoints) |
| `LAG` | Estimated phase lag |
| `AMPLITUDE` | Estimated amplitude |
| `OFFSET` | Estimated vertical offset |
| `MEAN_PERIODICITY` | Neural network periodicity score |
| `SCATTER` | Replicate scatter score |

---

## Arguments

| Argument | Default | Description |
|---|---|---|
| `--input_file` | *(required)* | Path to input `.tsv` file |
| `--output_dir` | *(required)* | Directory for output results |
| `--start_period` | `20.0` | Lower bound of period search range |
| `--end_period` | `28.0` | Upper bound of period search range |
| `--num_hidden` | `1000` | Number of hidden units in the network |
| `--act` | `splash` | Activation function type |
| `--train_itrs` | `10000` | Number of training iterations |
| `--batch_size` | `100` | Batch size during training |
| `--nnet_dir` | `./saved_nnets` | Directory to save/load trained models |
| `--retrain_binary` | `False` | Force retraining of the binary classifier |
| `--retrain_period` | `False` | Force retraining of the period estimator |
| `--lr` | `0.001` | Initial learning rate |
| `--lr_d` | `0.99996` | Learning rate decay per iteration |
| `--pval_comb` | `friston` | P-value combination method (`friston`, `edgington`, `fisher`, `stouffer`) |

---

## Project Structure

```
BioCycle/
├── BioCycle.py              # Main script
├── utils/
│   ├── data_utils.py        # Data generation utilities
│   ├── nnet_utils.py        # Neural network training/inference utilities
│   ├── nnet_models.py       # Model architecture definitions
│   ├── table_utils.py       # Input/output table parsing
│   └── misc_utils.py        # General helper functions
├── trajectory/
│   └── trajectory.py        # Trajectory data structure
├── null_dist/
│   └── mean_traj.pkl        # Precomputed null distribution for p-value calculation
├── requirements.txt
├── setup.sh
└── README.md
```
