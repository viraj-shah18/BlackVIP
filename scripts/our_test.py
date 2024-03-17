import os

DATA="./data/"
TRAINER="BLACKVIP"
SHOTS=16
SEED=1
CFG="vit_b16"
ptb="vit-mae-base"

DATASET="caltech101"
ep=20

spsa_os=1.0
alpha=0.4
spsa_a=0.01

b1=0.9
gamma=0.2
spsa_c=0.005
p_eps=0.2

opt_type='spsa-gc'

DIR=f"output/{DATASET}/{TRAINER}/{ptb}_{CFG}/shot{SHOTS}_ep{ep}/{opt_type}_b1{b1}/a{alpha}_g{gamma}_sa{spsa_a}_sc{spsa_c}_eps{p_eps}/seed{SEED}"
RESUME_DIR = DIR

prune_layer = ['p_trigger', 
               'conv_transpose_1',
               'conv_transpose_2', 'bn_1', 'act_1', 
               'conv_1', 'bn_2', 'act_2',
               'conv_transpose_3',
               'conv_transpose_4', 'bn_3', 'act_3',
               'conv_2', 'bn_4', 'act_4',
               'conv_transpose_5',
               'conv_transpose_6', 'bn_5', 'act_5',
               'conv_3', 'bn_6', 'act_6',
               'conv_transpose_7',
               'conv_transpose_8', 'bn_7', 'act_7',
               'conv_4', 'bn_8', 'act_8',
               'conv_transpose_9']

percentages = [0.4, 0.6, 0.8, 0.95, 0.99]

for layer in prune_layer:
    for percent in percentages:
        os.system(f"""
        python train.py \
        --root {DATA} \
        --seed {SEED} \
        --trainer {TRAINER} \
        --dataset-config-file configs/datasets/{DATASET}.yaml \
        --config-file configs/trainers/{TRAINER}/{CFG}.yaml \
        --output-dir {DIR} \
        --use_wandb \
        --wb_name blackvip \
        --eval-only \
        --load-epoch 20 \
        --resume {RESUME_DIR} \
        --prune-layer {layer} \
        --prune-percent {percent} \
        TRAIN.CHECKPOINT_FREQ 5 \
        DATASET.NUM_SHOTS {SHOTS} \
        DATASET.SUBSAMPLE_CLASSES all \
        OPTIM.MAX_EPOCH {ep} \
        TRAINER.BLACKVIP.PT_BACKBONE {ptb} \
        TRAINER.BLACKVIP.SPSA_PARAMS [{spsa_os},{spsa_c},{spsa_a},{alpha},{gamma}] \
        TRAINER.BLACKVIP.OPT_TYPE {opt_type} \
        TRAINER.BLACKVIP.MOMS {b1} \
        TRAINER.BLACKVIP.P_EPS {p_eps}
        """)