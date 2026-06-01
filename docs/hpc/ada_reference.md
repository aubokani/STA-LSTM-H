# Ada Lovelace HPC — operational reference

> Source: `sinfo`, `scontrol show partition`, `sacctmgr show assoc user=bokania`,
> `scontrol show config`, and `/etc/motd` captured **2026-04-27** on
> `ada.cqu.edu.au` (login node, RHEL 9.4). The public eResearch web pages
> (https://eresearch.cqu.org.au/high-performance-computing/) describe hardware
> in marketing terms but do **not** publish Slurm partitions, walltime caps, or
> array limits — this document is the ground truth for this project.

## Cluster identity

- **Hostname:** `ada.cqu.edu.au` (login node)
- **Scheduler:** Slurm (default Account `cquhpc`, default QOS `normal`)
- **Maintenance window:** 2026-06-03 (next scheduled outage per `/etc/motd`)
- **Support:** `eresearch@cqu.edu.au`

## Partitions (live `sinfo`, 2026-04-27)

| Partition       | Nodes | Hardware                 | GPUs/node           | Total GPUs | TimeLimit | Default |
|-----------------|------:|--------------------------|---------------------|-----------:|-----------|:-------:|
| `workq*`        | 10    | CPU only, 96 cores, ~770 GB RAM | —             | 0          | infinite  | ✅       |
| `gpuq`          | 2     | `hpc05-gln01–02`, 48 cores, 512 GB RAM | 1× **L40S 48 GB** | 2 | infinite  |         |
| `gpucomputeq`   | 2     | `hpc05-ghn01–02`, 48 cores, 512 GB RAM | 2× **H100 96 GB** | 4 | infinite  |         |

Important — the names are counter-intuitive:

- `gpuq`        → **L40S** (single GPU per node, 2 nodes total)
- `gpucomputeq` → **H100** (dual GPU per node, 4 GPUs total)

`PROJECT_PROGRESS.md` (and the prior `submit_hpc.sh`) labelled `gpucomputeq` as
"L40S nodes". That was wrong: `gpucomputeq` is the H100 partition.

`scontrol show partition` confirms: `gpucomputeq.Nodes=hpc05-ghn[01-02]`,
`gres/gpu=4`, and the corresponding `Gres` advertises `gpu:h100:2` per node.

## GresTypes

```
GresTypes = gpu, gpu:h100, gpu:l40
```

Use `--gres=gpu:1` for any GPU, or pin to a model:

```bash
#SBATCH --gres=gpu:l40:1     # force L40S
#SBATCH --gres=gpu:h100:1    # force H100
```

## User caps (account `cquhpc`, user `bokania`)

From `sacctmgr show assoc user=bokania`:

| Limit       | Value     |
|-------------|-----------|
| `MaxJobs`   | unlimited |
| `MaxSubmit` | unlimited |
| `MaxWall`   | unlimited |
| `MaxTRES`   | **`cpu=500`** |

Practical implication: with `-c 8` per task, the user can run ~62 tasks
concurrently before hitting the CPU cap. The 18-animal array fits comfortably.

## Cluster caps (`scontrol show config`)

| Setting          | Value |
|------------------|------:|
| `MaxArraySize`   | 1001  |
| `MaxJobCount`    | 10000 |
| `MaxStepCount`   | 40000 |
| `MaxTasksPerNode`| 512   |

The 18-animal array is well below all these.

## Walltime

All three partitions advertise `TimeLimit = infinite` and
`MaxTime = UNLIMITED`. The previous `-t 04:00:00` directive in `submit_hpc.sh`
was a self-imposed safety cap, not a cluster requirement. Keep it conservatively
to avoid runaways, but it can be increased if a long autoregressive evaluation
needs it.

## Submission policy notes

- Login node (`ada`) is for editing, submitting, and short tests only — do not
  run training there. Use Slurm interactive (`srun --pty`) or `sbatch`.
- Driver vs. PyTorch: login node reports CUDA driver `12.7`, but the project
  venv has `torch==2.11.0+cu130`. Login-node `torch.cuda.is_available()`
  returns False; this is expected. Compute nodes have the cu130-compatible
  driver.

## Software environment

- **Python module loaded by `setup_env.sh`:** `Python/3.12.3-GCCcore-13.3.0-deep-learning-cpu`
- **Project venv:** `~/projects/2026-STA-LSTM-H/.venv` (PyTorch `2.11.0+cu130`,
  pandas, scikit-learn, scipy, matplotlib, imbalanced-learn).

## Storage

`/home/bokania/projects/2026-STA-LSTM-H/` lives on the project NFS filesystem
(829 TB cluster-wide; **not backed up** per the eResearch site). The dataset
under `data/raw/` is 8.22 GB across 36 CSV files.

## Quick reference — typical job

```bash
#!/bin/bash
#SBATCH -J pilot-04
#SBATCH -p gpucomputeq           # H100 (or 'gpuq' for L40S)
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 02:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

cd "$SLURM_SUBMIT_DIR"
module purge
module load Python/3.12.3-GCCcore-13.3.0-deep-learning-cpu
source .venv/bin/activate
python src/run_animal.py --animal-id 4 --folds 5 --epochs 5 --max-horizon 30
```
