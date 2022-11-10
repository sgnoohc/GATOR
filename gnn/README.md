
## Command lines

For training one could do something like this:

    python train.py --seed 1234 --epochs 10 --hidden-size 200 --lr 0.00005 --save-model

For running inference on test set

    python inference.py trained_models/train_hiddensize200_PyG_LST_epoch10_0.8GeV_redo.pt
