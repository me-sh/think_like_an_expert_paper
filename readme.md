# Think like an expert paper (Neural alignment predicts learning outcomes, Nature Comms., 2021)

This repository contains code associated with the paper [_"Neural alignment predicts learning outcomes in students taking an introduction to computer science course_"](https://doi.org/10.1038/s41467-021-22202-3) by Meir Meshulam, Liat Hasenfratz, Hanna Hillman, Yun-Fei Liu, Mai Nguyen, Kenneth A. Norman and Uri Hasson. 

Imaging and behavioral data associated with this project is available on [_openNeuro.org_](https://openneuro.org/datasets/ds003233). 

The repository is organized as follows:

```
root
└── notebooks : jupyter notebooks
└── py : python code
└── masks : anatomical ROI and brain masks
```

### Instructions

After downloading the data folder from openNeuro, set the variable 'bids_path' in the code to point to the data folder.

Use notebooks for pre-processing of raw data (requires FSL; dependencies in py folder), behavioral analysis and ROI analysis. Analysis notebooks contain the expected outputs. Run times for a single analysis on a single region of interest (ROI) are <1h on a single CPU core.

Use similarity_searchlight.py for whole-brain analysis (requires BrainIAK searchlight).

The code was tested under GNU/Linux (x86_64 architecture) with Jupyter Notebook and BrainIAK (version information below). No special installation is required.

[ Python ](https://www.python.org/) v. 3.7.4

[ Jupyter Notebook ](https://jupyter.org/)  v. 6.0.2

[ BrainIAK ](https://github.com/brainiak)  v. 0.9.1

[ FSL ](https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/) v. 6.0.1

