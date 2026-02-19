# pyMSHBM Pipeline Guide

A Python implementation of the Multi-Scale Hierarchical Bayesian Model (MS-HBM) for individual-specific brain parcellation from resting-state fMRI data.

**References:**

- Kong et al. (2019). "Spatial Topography of Individual-Specific Cortical Networks Predicts Human Cognition, Personality, and Emotion." *Cerebral Cortex* 29(6):2533-2551.
- Xue et al. (2021). "The Detailed Organization of the Human Cerebellum Estimated by Intrinsic Functional Connectivity Within the Individual." *Journal of Neurophysiology*.

## Installation

```bash
uv sync
```

Requires Python >= 3.13. Dependencies: `numpy`, `scipy`, `nibabel`, `h5py`.

## Pipeline Overview

The pipeline has six steps:

1. **Load fMRI data** and compute functional connectivity (FC) profiles
2. **Average profiles** across subjects
3. **Initialize clusters** via von Mises-Fisher (vMF) clustering
4. **Estimate group priors** from training subjects (hierarchical Bayesian EM)
5. **Generate individual parcellations** for held-out subjects
6. **(Optional) Cerebellar parcellation** via winner-take-all

Steps 1-3 prepare the data. Step 4 is the main training step. Step 5 applies the trained model. Step 6 extends cortical parcellations to the cerebellum.

## Step 1: Compute FC Profiles

FC profiles capture the functional connectivity fingerprint of each vertex/voxel. You compute Pearson correlations between target vertices and seed regions, then apply a Fisher-Z transform.

### Loading fMRI data

```python
from pymshbm.io.readers import read_fmri

bundle = read_fmri("sub-01_ses-01.dtseries.nii")
data = bundle.series  # shape: (N_vertices, N_timepoints)
```

Supported formats:
- `.dtseries.nii` (CIFTI)
- `.nii` / `.nii.gz` (NIfTI)
- `.mat` (must contain a `profile_mat` variable)

### Computing profiles

```python
from pymshbm.core.profiles import generate_profiles

# targ_data: (T, N_targ) - time series for target vertices (e.g., cortex)
# seed_data: (T, N_seed) - time series for seed ROIs
profile = generate_profiles(
    targ_data=targ_data,
    seed_data=seed_data,
    censor=censor_mask,  # optional (T,) boolean; True = keep frame
)
# profile shape: (N_targ, N_seed) - Fisher-Z correlations
```

Repeat this for every subject and session. The resulting profiles should be L2-normalized (unit norm per row) before feeding into the training pipeline.

### Assembling the data array

Stack all profiles into a single 4D array:

```python
import numpy as np

# profiles[s][t] is the (N, D) profile for subject s, session t
# N = number of vertices, D = number of seed ROIs
N, D = profiles[0][0].shape
S = len(profiles)          # number of subjects
T = len(profiles[0])       # number of sessions per subject

data = np.zeros((N, D, S, T), dtype=np.float64)
for s in range(S):
    for t in range(T):
        row = profiles[s][t]
        norms = np.linalg.norm(row, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        data[:, :, s, t] = row / norms  # L2-normalize each vertex
```

## Step 2: Average Profiles Across Subjects

```python
from pymshbm.core.profiles import avg_profiles

# Flatten all subject/session profiles into a list
all_profiles = [profiles[s][t] for s in range(S) for t in range(T)]
avg = avg_profiles(all_profiles)  # shape: (N, D)
```

## Step 3: Initialize Clusters

Run vMF clustering on the group-averaged profile to get initial network centroids:

```python
from pymshbm.core.profiles import generate_ini_params

num_networks = 17  # e.g., 17 networks

labels, centroids = generate_ini_params(
    avg_profile=avg,       # (N, D) from step 2
    num_clusters=num_networks,
    num_init=1000,         # number of random restarts
    out_dir="output/",     # saves output/group/group.mat (optional)
)
# labels: (N,) 1-indexed cluster labels (0 = medial wall)
# centroids: (D, num_networks) unit-norm cluster centers
```

The `centroids` matrix is the `g_mu` initialization for training.

## Step 4: Estimate Group Priors (Training)

This is the main model fitting step. It runs a hierarchical Bayesian EM algorithm that separates inter-subject from intra-subject variability.

```python
from pymshbm.pipeline.training import params_training

params = params_training(
    data=data,                 # (N, D, S, T) from step 1
    g_mu=centroids,            # (D, L) from step 3
    num_clusters=num_networks,
    max_iter=50,               # max EM iterations
    conv_th=1e-5,              # convergence threshold
    output_dir="output/",      # saves output/priors/Params_Final.mat
    subject_ids=["sub-01", "sub-02", ...],  # optional
    save_all=True,             # retain per-subject estimates
)
```

### Training outputs

When `output_dir` is specified, the pipeline saves `output/priors/Params_Final.mat` containing:

| Field | Shape | Description |
|-------|-------|-------------|
| `mu` | (D, L) | Group-level connectivity profile per network |
| `epsil` | (1, L) | Inter-subject variability concentrations |
| `sigma` | (1, L) | Intra-subject variability concentrations |
| `theta` | (N, L) | Spatial prior (network probability per vertex) |
| `kappa` | (1, L) | Session-level concentration parameters |
| `iter_inter` | scalar | Number of iterations completed |
| `Record` | (1, iter) | Cost function values per iteration |

When `save_all=True`, the output also includes `s_psi`, `s_t_nu`, and `s_lambda` (per-subject/session estimates), and individual parcellations are saved to `output/ind_parcellation/`.

### Dimension constraints

The profile dimension `D` determines the initial concentration parameter:
- D < 1201: `ini_concentration = 500`
- 1200 <= D < 1800: `ini_concentration = 650`
- D >= 1800: not supported

### Monitoring convergence

Check the cost function record to verify convergence:

```python
import matplotlib.pyplot as plt

plt.plot(params.record)
plt.xlabel("Iteration")
plt.ylabel("Cost")
plt.title("Group Prior Convergence")
```

## Step 5: Generate Individual Parcellations

Apply trained group priors to generate a parcellation for a new (held-out) subject.

### Loading pre-trained priors

```python
from pymshbm.io.mat_interop import load_params_final

group_priors = load_params_final("output/priors/Params_Final.mat")
```

Or pass the path directly to `parcellation_single_subject` (it loads automatically).

### Running parcellation

```python
from pymshbm.pipeline.single_subject import parcellation_single_subject

lh_labels, rh_labels = parcellation_single_subject(
    data=subject_data,          # (N, D, 1, T) - single subject, normalized
    group_priors=group_priors,  # MSHBMParams or path to Params_Final.mat
    neighborhood=neighborhood,  # (N, max_neighbors) adjacency matrix
    w=200.0,                    # spatial prior weight
    c=50.0,                     # MRF smoothness weight
    max_iter=50,
)
# lh_labels: (N/2,) integer labels for left hemisphere
# rh_labels: (N/2,) integer labels for right hemisphere
```

### Key parameters

- **`w`** (spatial prior weight): Controls how much the parcellation follows the group spatial prior (`theta`). Higher values produce parcellations closer to the group average. Typical range: 60-200.
- **`c`** (MRF smoothness weight): Controls spatial contiguity of parcels. Higher values produce smoother, more contiguous regions. Typical range: 30-100.

### Neighborhood matrix

The `neighborhood` array defines spatial adjacency between vertices. Shape is `(N, max_neighbors)` where each row lists the indices of neighboring vertices. Use `-1` for padding when a vertex has fewer neighbors than `max_neighbors`. This is determined by your surface mesh (e.g., fsaverage5, fs_LR_32k).

### Input data format

The single-subject data array must be:
- Shape `(N, D, 1, T)` where `T` is the number of sessions
- L2-normalized (unit norm per vertex row)
- The third dimension is 1 because there is only one subject

## Step 6: Validate Parcellation Quality

Use homogeneity to evaluate how well vertices within each parcel share similar FC profiles:

```python
from pymshbm.core.validation import parameters_validation

labels = np.concatenate([lh_labels, rh_labels])
score = parameters_validation(
    labels=labels,   # (N,) integer labels
    data=fc_data,    # (N, D) FC profiles for this subject
)
print(f"Homogeneity: {score:.4f}")  # higher is better
```

Use this metric to tune `w` and `c` on a validation set before applying to test subjects.

## Step 7 (Optional): Cerebellar Parcellation

Extend cortical parcellations to the cerebellum using a winner-take-all approach based on cerebellum-to-cortex functional connectivity:

```python
from pymshbm.cerebellum.parcellation import cerebellum_parcellation

labels, confidence = cerebellum_parcellation(
    surf_labels=cortical_labels,  # (N,) cortical parcellation
    vol2surf_fc=fc_matrix,        # (M, N) cerebellum-cortex FC
    top_x=100,                    # top correlated vertices to consider
)
# labels: (M,) cerebellar parcel labels
# confidence: (M,) confidence scores per voxel
```

## Complete Example

```python
import numpy as np
from pymshbm.core.profiles import generate_profiles, avg_profiles, generate_ini_params
from pymshbm.pipeline.training import params_training
from pymshbm.pipeline.single_subject import parcellation_single_subject

# --- Data preparation ---
# Load and compute FC profiles for each subject/session
# (details depend on your data organization)
# Result: profiles[subject][session] = (N, D) array

# Assemble into 4D array
N, D = profiles[0][0].shape
S, T = len(profiles), len(profiles[0])
data = np.zeros((N, D, S, T))
for s in range(S):
    for t in range(T):
        row = profiles[s][t]
        norms = np.linalg.norm(row, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        data[:, :, s, t] = row / norms

# --- Initialization ---
all_profiles = [profiles[s][t] for s in range(S) for t in range(T)]
avg = avg_profiles(all_profiles)
labels, centroids = generate_ini_params(avg, num_clusters=17, num_init=1000)

# --- Training ---
params = params_training(
    data=data,
    g_mu=centroids,
    num_clusters=17,
    output_dir="output/",
    save_all=True,
)

# --- Individual parcellation (new subject) ---
lh, rh = parcellation_single_subject(
    data=new_subject_data,  # (N, D, 1, T)
    group_priors="output/priors/Params_Final.mat",
    neighborhood=neighborhood,
    w=200.0,
    c=50.0,
)
```

## File I/O Utilities

### Loading `.mat` files

```python
from pymshbm.io.mat_interop import load_mat, save_mat
from pymshbm.types import FileFormat

data = load_mat("file.mat")          # auto-detects v5/v7.3/npz
save_mat("out.mat", data, fmt=FileFormat.MAT_V5)
```

### Converting label files

```python
from pymshbm.pipeline.label_convert import label2cifti

results = label2cifti("output/ind_parcellation/")
# returns dict: filename -> (lh_labels, rh_labels)
```

## Supported Surface Spaces

- fsaverage5 (10,242 vertices per hemisphere)
- fsaverage6 (40,962 vertices per hemisphere)
- fs_LR_32k (32,492 vertices per hemisphere)
- fs_LR_164k
- Any custom mesh (provide your own neighborhood matrix)
