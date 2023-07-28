# GATOR ML pipeline
This code implements a pipeline for training and testing Graph Neural Networks (GNNs) and Deep Neural Networks (DNNs) on LST data. 
A set of scripts has also been developed for deploying these trainings via `slurm` on the [HiPerGator](https://www.rc.ufl.edu/about/hipergator/) (HPG) system at the University of Florida--these scripts may work on other systems managed by `slurm`, but no significant effort has been made to ensure this.
The code is organized as follows:
- `bin`: contains executables for managing and monitoring GATOR ML jobs
- `python`: contains the scripts responsible for running the entire ML pipeline, from data ingress, to training, to saving the model inferences; these scripts are expected to be well-maintained and generalizeable to all GATOR ML models
- `scripts`: contains various scripts for doing more specific operations; e.g. plotting or creating configs for hyperparameter scans
- `configs`: contains configuration JSONs for each GATOR ML model

## Quick start
1. Start an interactive session on a HPG GPU node and source the setup script
```
srun --partition=gpu --gpus=1 --mem=16gb --constraint=a100 --pty bash -i
source setup_hpg.sh
```
2. Run each step of the pipeline for an example config
```
python python/ingress.py configs/ChangGNN_MDnodes_LSedges.json
python python/train.py configs/ChangGNN_MDnodes_LSedges.json
python python/infer.py configs/ChangGNN_MDnodes_LSedges.json
python scripts/plot.py configs/ChangGNN_MDnodes_LSedges.json
```
3. Rather than running the above interactively, this can instead be run via a batch job
```
./bin/submit configs/ChangGNN_MDnodes_LSedges.json
```

## Specific instructions
### Model configuration
All GATOR ML models are configured via a JSON. 
This JSON can be slightly different for GNNs vs. DNNs, so each

#### GNN configuration
- `base_dir`: base directory for all outputs (on HPG, this should be somewhere on `/blue`)
- `model`:
  - `name`: name of the model to use; check `python/models.py` for the available models
  - `n_hidden_layers`: number of hidden layers for GNN MLPs,
  - `hidden_size`: size of hidden layers of GNN MLPs,
  - `message_size`: size of message (default: number of edge features),
  - `latend_node_size`: size of latent node (default: number of node features),
  - `n_message_passing_rounds`: number of message passing rounds,
  - `message_aggregator`: message aggregator (default: "max")
- `ingress`:
  - `inputs_files`: list of input ROOT files (assumes truth labels are saved in each file)
  - `ttree_name`: name of the input TTree (assumes the name is the same across all input files)
  - `undirected`: boolean specifying whether or not graph is undirected
  - `edge_indices`: list containing two branches containing the inner and outer node indices respectively
  - `edge_features`: list of branches containing edge features (e.g. LS pT, eta, phi, ...)
  - `node_features`: list of branches containing node features (e.g. MD pT, x, y, z, ...)
  - `transforms`: sub-mapping of `feature`:`transform` key-value pairs; supported transformations can be found in `python/ingress.py` in the `transform` function
  - `plot_labels`: sub-mapping of `feature`:`label` key-value pairs for plotting; labels are passed directly to `matplotlib.axes.Axes.set_xlabel`
  - `truth_label`: name of branch containing the truth label for the intended classification task
  - `branch_filter`: filter that selects only relevant branch names (optional)
- `train`:
  - `train_frac`: fraction of events to use for training (the rest is used for testing)
  - `learning_rate`: learning rate,
  - `seed`: seed for random number generation
  - `scheduler_name`: name of pytorch scheduler to use
  - `scheduler_kwargs`: keyword arguments for the specified scheduler
  - `n_epochs`: number of epochs to train for (no early stopping)

#### DNN configuration
- `base_dir`: base directory for all outputs (on HPG, this should be somewhere on `/blue`)
- `model`:
  - `name`: name of the model to use; check `python/models.py` for the available models (only models whose names contain "DNN" are allowed)
  - `n_hidden_layers`: number of hidden layers,
  - `hidden_size`: size of hidden layers,
- `ingress`:
  - `inputs_files`: list of input ROOT files (assumes truth labels are saved in each file)
  - `ttree_name`: name of the input TTree (assumes the name is the same across all input files)
  - `undirected`: boolean specifying whether or not graph is undirected
  - `edge_indices`: list containing two branches containing the inner and outer node indices respectively
  - `edge_features`: list of branches containing edge features (e.g. LS pT, eta, phi, ...); DNN input features are the edge and node features concatenated together
  - `node_features`: list of branches containing node features (e.g. MD pT, x, y, z, ...); DNN input features are the edge and node features concatenated together
  - `transforms`: sub-mapping of `feature`:`transform` key-value pairs; supported transformations can be found in `python/ingress.py` in the `transform` function
  - `plot_labels`: sub-mapping of `feature`:`label` key-value pairs for plotting; labels are passed directly to `matplotlib.axes.Axes.set_xlabel`
  - `truth_label`: name of branch containing the truth label for the intended classification task
  - `branch_filter`: filter that selects only relevant branch names (optional)
- `train`:
  - `train_frac`: fraction of events to use for training (the rest is used for testing)
  - `learning_rate`: learning rate,
  - `seed`: seed for random number generation
  - `scheduler_name`: name of pytorch scheduler to use
  - `scheduler_kwargs`: keyword arguments for the specified scheduler
  - `n_epochs`: number of epochs to train for (no early stopping)

### Submitting and monitoring batch jobs
The entire pipeline described in the "Quick start" instructions can be run with `bin/submit` as follows:
```
./bin/submit configs/CONFIG.json
```
Moreover, several jobs can be started for all configs matching a wildcard. 
Suppose the following configs have been written: `configs/cfg_1.json`, `configs/cfg_2.json`, `configs/cfg_3.json`.
Then, jobs for all three may be submitted at once:
```
./bin/submit configs/cfg_*.json
```
A subset of steps may also be run by specifying each step as flags in the above command. 
For example, just the ingress and training step may be run as follows:
```
./bin/submit configs/CONFIG.json --ingress --train
```
Some job properties may also be set via the CLI, like the time limit and account/QOS properties. 
The full set of options can be found by checking the `--help` output:
```
$ ./bin/submit --help
usage: submit [-h] [--ingress] [--train] [--infer] [--plot] [-t TIME] [--account ACCOUNT] [--qos QOS] [config_jsons ...]

Submit batch jobs

positional arguments:
  config_jsons          config JSON

options:
  -h, --help            show this help message and exit
  --ingress             run the data ingress step
  --train               run the training step
  --infer               run the inference step
  --plot                run the plotting step
  -t TIME, --time TIME  same as sbatch --time
  --account ACCOUNT     same as sbatch --account
  --qos QOS             same as sbatch --qos
```
Finally, jobs may be monitored with the `bin/queue` executable:
```
$ ./bin/queue
Active jobs:
/path/to/cfg_1/slurm-JOBID.out
/path/to/cfg_2/slurm-JOBID.out
Inactive jobs:
(slurm id: JOBID) cfg_3
```
From the above output, one can find the `slurm` log files for each active job, as well as a list of inactive jobs. 
One can then use additional `slurm` commands to diagnose issues with a given job.