import os
from pathlib import Path

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

DIR_PATH = Path("output", DATASET, TRAINER, f"{ptb}_{CFG}", f"shot{SHOTS}_ep{ep}", f"{opt_type}_b1{b1}", f"a{alpha}_g{gamma}_sa{spsa_a}_sc{spsa_c}_eps{p_eps}", f"seed{SEED}")
RESUME_DIR = DIR_PATH


os.system(f"""
python train.py \
--root {DATA} \
--seed {SEED} \
--trainer {TRAINER} \
--dataset-config-file configs/datasets/{DATASET}.yaml \
--config-file configs/trainers/{TRAINER}/{CFG}.yaml \
--output-dir {DIR_PATH} \
--eval-only \
--load-epoch 20 \
--resume {RESUME_DIR} \
--prune-experiment \
--use_wandb \
--wb_name blackvip \
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