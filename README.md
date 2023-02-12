
## Running on HiPerGator

### Instructions

Login to hpg

    ssh $user@hpg.rc.ufl.edu

On login[1-6] node checkout the code

    git clone git@github.com:sgnoohc/GATOR.git

Then fire up an interactive (option -i) SLURM job. (SLURM is an alternative to Condor)
Following command requests one A100 GPU node.

    srun --partition=gpu --gpus=1 --mem=16gb --constraint=a100 --pty bash -i

Within the node:

    cd GATOR/gnn
    source setup_hpg.sh
    python graphdata.py
    python train.py
    python inference.py
    make -j
    ./writetree

This creates the `output.root` that contains the MD and LS with `LS_score` branch

### Input

The beginning input ```LSTNtuple.root``` is located in Philip's home area.
The intermediate values are also in the home area.
In the python scripts in various places, the respective intermediate results outputs are located in the path noted in the comment. 

### Displaying via plotly

We will use jhub.rc.ufl.edu

In the terminal install plotly / uproot / uproot-awkward

    module load python/3.10
    pip install plotly
    pip install uproot
    pip install awkward-pandas

Then go to `display/` and open EventDisplay.ipynb for examples

