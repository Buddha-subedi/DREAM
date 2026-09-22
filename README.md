[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Buddha-subedi/DREAM/blob/main/DREAM_demo.ipynb)
# An Attention-based Conditional Diffusion Model for Super-resolved Passive Microwave Satellite Precipitation Retrievals
This repository introduces **D**iffusion-based PMW **R**ainfall r**E**trievals with **A**ttention **M**echanisms (DREAM), a two-stage probabilistic framework for super-resolved PMW retrieval at near-global scale. First, a residual U-Net (Res-UNet) trained with mean squared error (MSE) and sliced Wasserstein distance loss generates coarse deterministic estimates at Global Precipitation Measurement (GPM) Microwave Imager (GMI) resolution. Second, a conditional denoising diffusion probabilistic model (DDPM) learns the residual toward high-resolution Dual-frequency Precipitation Radar (DPR) targets, fusing multi-frequency brightness temperatures and reanalysis variables through channel, cross-, self-, and bottleneck dual-attention mechanisms. Evaluated on 2023 GPM overpasses including Hurricane Calvin, DREAM substantially outperforms deterministic baselines in spatial variability, extreme precipitation frequency, and power spectral fidelity.

<p align="center">
  <img src="Figures/Fig_01.png" width="1000" />
</p>

<p align="center"><em>Schematic of the DREAM architecture. (a) A Res-UNet generates deterministic rainfall retrievals at the native radiometric resolution, while an attention-enhanced conditional diffusion model learns and generates rainfall residuals relative to DPR observations. (b) Forward and reverse diffusion processes. (c) U-Net backbone. (d) Cross-attention module. (e) Channel-attention module. (f) Dual-attention bottleneck combining cross- and channel-attention to fuse GMI brightness temperatures and ERA5 variables across multiple spatial scales.</em></p>

<a name="4"></a> <br>
## Code

<a name="41"></a> <br>
###   Setup
To run this notebook on Google Colab, clone this repository
```python
!git clone https://github.com/Buddha-subedi/DREAM.git
os.chdir("PMWPrecip_TLP-R2S")
```


```python
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from pathlib import Path
import xgboost as xgb
import os
import scipy.io
import pmw_utils
import importlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, mean_squared_error
import scipy.stats as stats
from scipy.interpolate import interp1d
importlib.reload(pmw_utils)
from pmw_utils import plot_confusion_matrix, TLPR2S_model
```
<a name="42"></a> <br>
 ### Load the Data
 
```python
paths_phase = {
    'cpr_train': 'data/df_cpr_phase_train.npz',
    'cpr_test':  'data/df_cpr_phase_test.npz',
    'dpr_train': 'data/df_dpr_phase_train.npz',
    'dpr_test':  'data/df_dpr_phase_test.npz',
    'era5_train': 'data/df_era5_phase_train.npz',
    'era5_test':  'data/df_era5_phase_test.npz'
}
data = {k: np.load(p) for k, p in paths_phase.items()}
dfs = {k: pd.DataFrame(dict(v)) for k, v in data.items()}
df_cpr_phase_train = dfs['cpr_train']
df_cpr_phase_test  = dfs['cpr_test']
df_dpr_phase_train = dfs['dpr_train']
df_dpr_phase_test  = dfs['dpr_test']
df_era5_phase_train = dfs['era5_train']
df_era5_phase_test  = dfs['era5_test']
```



<a name="43"></a> <br>
 ### Train the TLP-R2S Model
TLP-R2S Model has 3 base learners. The hyperparameters and snippet of code adopted for stage 1 and stage 2 for the phase detection is provided below

```python
#stage 1
classes = np.unique(df_era5_phase_train['Prcp flag'])
class_weights = {0: 1, 1: 1.15, 2: 1.32}

sample_weights_70 = df_era5_phase_train['Prcp flag'].map(lambda x: class_weights[classes.tolist().index(x)])

params = {
    'objective': 'multi:softmax',
    'num_class': 3,
    'eval_metric': 'merror',
    'reg_alpha': 1.351,
    'reg_lambda': 5.219,
    'max_depth': 14,
    'num_parallel_tree': 3,
    'learning_rate': 0.41302,
    'gamma': 0.225,
    'verbosity': 0
}

booster_era5 = xgb.train(
    params=params,
    dtrain=dtrain,
    evals=evals,
    num_boost_round=88,
    verbose_eval=True
)


#stage 2
classes = np.unique(df_phase_train['Prcp flag'])
class_weights = {0: 1, 1: 1.267, 2: 1.966}
sample_weights_sat = df_phase_train['Prcp flag'].map(lambda x: class_weights[classes.tolist().index(x)])

# Set parameters
params_1 = {
    'objective': 'multi:softmax',
    'num_class': 3,
    'eval_metric': 'merror',
    'subsample': 0.5,
    'reg_alpha': 6.948,
    'reg_lambda': 5.0278,
    'max_depth': 16,
    'num_parallel_tree': 6,
    'learning_rate': 0.011,
    'gamma': 0.32,
    'verbosity': 0
}

booster_era5 = xgb.train(
    params=params_1,
    dtrain=dtrain_era5,
    evals=evals,
    num_boost_round=83,
    verbose_eval=True
)

# Train with the new data (booster here is the final model that is first trained on coarse
# resolution information from ERA5 and then fine-tuned on fine resolution satellite information)
params_2 = {
    'objective': 'multi:softprob',
    'num_class': 3,
    'eval_metric': 'merror',
    'reg_alpha': 6.948,
    'reg_lambda': 5.0278,
    'max_depth': 15,
    'num_parallel_tree': 6,
    'learning_rate': 0.018,
    'gamma': 0.32,
    'verbosity': 0
}

booster_cpr = xgb.train(
    params_2,
    dtrain_cpr,
    num_boost_round=80,
    evals=evals,
    xgb_model=booster_era5,
    verbose_eval=True,
    feval=f1_eval_all_classes
)
```


<a name="44"></a> <br>
 ### Orbital Retrievals
```python
[phase, rain, snow, latitude, longitude] = TLPR2S_model(path_orbit_004780, booster, snow_rate_booster, rain_rate_booster, df_cdf_rain, df_cdf_snow);
```
<p align="center">
  <img src="images/Fig_04.png" alt="Training for ERA5-CPR classifier base learner" width="700" />
</p>
<p align="center">
  <em>Three selected GMI TBs (a--c) and precipitation from MRMS (d), TLP-R2S (e), and GPROF (f) for orbit 045821 on March 23, 2022, over the Midwest United States. Likewise, selected GMI TBs (g--i) and corresponding MRMS (j), TLP-R2S (k), and GPROF (l) precipitation for orbit 045212 on February 11, 2022, over Colorado and Wyoming.</em>
</p>



## Dataset
The complete dataset for training the networks and retrieving sample orbits is available here: (https://drive.google.com/drive/folders/1NwouPlF4kF2kdHWRwpCHfHnjyaW14xzP?usp=sharing).
