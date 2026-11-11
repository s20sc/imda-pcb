# IMDA for PCB inspection

IMDA encodes exposure-timed camera motion as quadratic trajectory maps for image restoration. This repository accompanies the controlled PCB inspection study.

- `pcb/code/revision/` implements motion synthesis, PCB preprocessing, float-map training, inspection metrics, and paired statistics.
- `pcb/tools/visa_source.py` checks the VisA source split.
- `gdn/code/` implements the start-reference camera-motion field, the GyroDeblurNet adapter, and the paired GDN analysis.
- The protocol files specify the motion and comparison settings.

Install the CPU dependencies with `python -m pip install -r requirements.txt`. Run `python pcb/code/revision/imu_synthesis.py --help` for the standalone synthesis interface. The motion JSON specifies `K`, `times_seconds`, and `angular_rates_rad_s`. Use calibrated pinhole intrinsics for the supplied image size and camera-frame angular rates in radians per second.

Training additionally requires PyTorch, the allocated dataset, and the upstream model files. Obtain the GAMD directory from [GAMD-Net at bcacee1](https://github.com/lsmlovefm/GAMD-Net/tree/bcacee1e39ae407e5847ea872a9c27b095b05521) and place it in `pcb/code/revision/`. For GDN, place the pinned `model.py` and `arch_util.py` from [GyroDeblurNet at ff8beec](https://github.com/hmyang0727/GyroDeblurNet/tree/ff8beec5d0af6e8e132bbff1db19ce46247ce869) in `gdn/vendor/official/` and install its native model dependencies.

The complete training and evaluation runs use additional allocation files, checkpoint files, and execution records. The paper's separate reproduction supplements provide recorded predictions and their statistical replay. This repository provides the source components; the datasets, weights, machine-specific run plans, and result archives are distributed separately or obtained from their original sources.

The synthesis CLI and CPU metric functions work with the checkout and the listed CPU dependencies. The formal PCB statistics CLI requires the original training plan, completed producer records, and source manifests. The GDN statistics module likewise uses the pinned PCB statistics tree and completed GDN records. To reproduce the reported aggregate results from the study supplements, run `python replay_statistics.py` in the extracted PCB supplement and `python replay_gdn_statistics.py` in the extracted GDN supplement, using the dependencies supplied with each. These commands replay arithmetic on recorded predictions.
