# Effectiveness of Federated Unlearning Algorithms

This repository contains the code used to evaluate membership inference and
unlearning techniques in federated settings.  Experiments are executed using
[Flower](https://flower.dev/) for orchestration and TensorFlow for model
training.

## Data
The experiments are based on the **Texas-100** dataset.  A helper script
(`src/prepare_dataset.py`) downloads the original data, samples 30,000 records
with a fixed seed and stores an 80/20 train-test split in
`data/texas100_subset.npz`.

```bash
python src/prepare_dataset.py --root-dir data
```

## Setup
The easiest way to prepare the environment is with Ansible:

```bash
ansible-playbook -i "localhost," -c local ansible/environment_setup.yml
source ~/venv/bin/activate
```

This creates a Python 3.11 virtual environment and installs the required
packages:
* `flwr[simulation]`
* `tensorflow`
* `numpy`
* `scikit-learn`
* `matplotlib`

Hardware requirements are modest: a CPU with ~8 GB RAM is sufficient for small
experiments.  GPUs (see `scripts/gpu_job.sbatch`) accelerate training but are
optional.

## Usage
### Local quick test
After setup and dataset preparation, a single experiment can be executed
locally using Flower’s simulation mode:

```bash
python src/experiment_runner.py data/texas100_subset.npz
```

### Federated training
To launch a standalone federated run, start one server and multiple clients in
separate terminals:

```bash
# Terminal 1
python src/run_federated.py --role server --address 127.0.0.1:8080

# Terminal 2..N
python src/run_federated.py --role client --cid <CLIENT_ID> --address 127.0.0.1:8080 --data-dir data/clients
```

### SLURM jobs
For large-scale experiments on a SLURM cluster, submit the provided scripts:

```bash
# CPU-only job
sbatch scripts/cpu_job.sbatch

# Single-GPU job
sbatch scripts/gpu_job.sbatch
```

Both scripts assume the virtual environment created above is available at
`~/venv`.

## Reproducing thesis results
Prepare the dataset, install the environment, and then submit the appropriate
SLURM script (`cpu_job.sbatch` or `gpu_job.sbatch`).  Logs and model checkpoints
are written to the `logs/` directory.  Ensure the repository is tagged (e.g.,
`git tag -a v0.1 -m "archive"`) to make an immutable snapshot of the code.

## License
This project is licensed under the MIT License; see [LICENSE](LICENSE).