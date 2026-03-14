# OmniFold Unfolding Standardization

The standardization of ML-based unfolding results for
publication, preservation, and reuse in high-energy physics experiments.

Modern unfolding methods like OmniFold produce per-event weights rather than
fixed binned histograms. While this enables flexible reinterpretation, it also
raises challenges for publication, reproducibility, and long-term usability.

---

## Goals

- Analyze the given dataset.
- Define what results should be published when unfolding is performed using OmniFold.
- Design a standardized format for per-event weights and accompanying metadata.

---

## Repository Structure
```
├── weighted_histogram.py   # Weighted histogram function with uncertainty bands + tests (TASK 3)
├── metadata.yaml           # Metadata schema for OmniFold unfolding results (TASK 2)
├── schema_design.md        # Justification and design notes for the metadata schema (TASK 2)
├── gap_analysis.md         # Analysis of information gaps in current OmniFold outputs (TASK 1)
├── testing.ipynb           # Exploratory analysis notebook (Analysis used for TASK 1 AND TASK 2)
└── data/                   # Not included — see below
```

---

## Components

### Metadata Schema (`metadata.yaml`)
A YAML schema designed to accompany OmniFold HDF5 weight files, capturing:
- Dataset identity and experiment context
- Monte Carlo simulation provenance
- Event selection criteria
- Full column-level documentation for all weight types
- OmniFold training configuration and software environment

### Weighted Histogram Tool (`weighted_histogram.py`)
A self-contained Python function for computing and plotting weighted histograms
from OmniFold per-event weights. it's functionality include:
- Per-bin weight summation (not event counting)
- Optional statistical uncertainty bands from ensemble weights
- edge case handling (NaN, negative weights, empty ranges)
- 14 tests checking for critical failure

---

## how to run this?
```bash
git clone https://github.com/yourusername/OmnifoldUnfoldingAndHistogram.git
cd OmnifoldUnfoldingAndHistogram
pip install numpy matplotlib pytest pandas
```

### Running the tests
```bash
pytest weighted_histogram.py -v
```

---

## Data

The HDF5 data files are not included in this repository due to file size.

Three HDF5 files containing pre-calculated OmniFold weights for pseudo-data, that mimics a real ATLAS measurement:

multifold.h5 - the nominal result, based on MG5 simulation 
multifold_sherpa.h5 - an alternative generator (Sherpa), treated as a systematic uncertainty 
multifold_nonDY.h5 - an alternative sample composition (includes EW Zjj/VBF and diboson), treated as another systematic uncertainty 
These files can be downloaded from this Zenodo link(https://zenodo.org/records/11507450). Be sure to work with the files under files/pseudodata.

Download the files and place them in a `data/` folder at the root of the project.

Expected structure:
```
data/
  multifold.h5            # Nominal result (MadGraph5, 418k events)
  multifold_sherpa.h5     # Generator variation (Sherpa, 326k events)
  multifold_nonDY.h5      # Sample composition variation (326k events)
```

---
