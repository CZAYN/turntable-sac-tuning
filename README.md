# Turntable SAC Tuning

## Background

This project uses Soft Actor-Critic (SAC) to automatically tune the current, speed, and position control loops of a turntable servo system in a physics simulation of the motor. It optimizes 11 parameters across three PIDF controllers and the speed-loop disturbance observer-based controller (DOBC), using bandwidth, stability margins, and step-response metrics to evaluate performance.

The project is intended for simulation research. Applying the tuned parameters to real hardware requires system identification and further validation.

## Quick Start

The examples below use Windows PowerShell and Conda. Install Git and Conda first. The environment file specifies Python 3.11 and a CUDA build of PyTorch.

### 1. Clone the repository and install the environment

```powershell
git clone https://github.com/CZAYN/turntable-sac-tuning.git
cd turntable-sac-tuning

conda env create -f environment-local.yml
conda activate elc-rl-local
python -m pip install --no-deps --no-build-isolation -e .
```

Keep the environment activated and run all remaining commands from the repository root.

### 2. Check the environment

```powershell
python -m pip check
python scripts/check_tuning_env.py --quick
```

These commands check dependencies, simulation interfaces, and SAC policy construction without starting training. The environment report is saved to `outputs/environment_validation_physics.json`.

### 3. Run training

After the checks pass, start a training run with one random seed:

```powershell
python scripts/train_sac.py --device cuda --n-envs 2 --output-dir outputs/sac_run_01
```

The command uses [config/sac_training.json](config/sac_training.json) and runs four training stages in sequence: current, speed, position, and joint optimization. This starts a full training run. If CUDA is unavailable, replace `--device cuda` with `--device cpu`.

Models, checkpoints, and reports are saved to `outputs/sac_run_01/`, with TensorBoard logs in its `tensorboard/` subdirectory. Use a new or empty output directory for each experiment. Models and checkpoints from earlier training protocols are incompatible.

---

Further documentation (in Chinese): [Workflow and Algorithm Design](PROJECT_WORKFLOW.md) | [Local Setup and Checks](LOCAL_SETUP.md) | [Friction Parameter Identification](docs/LUGRE_IDENTIFICATION.md)
