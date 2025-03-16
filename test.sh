export WANDB_API_KEY=7352f9a349b74e01062672d0bc0bd3a8094677e2

OUTPATH=/lpai/output/models/test
BS=512
LR=1e-3
N_EPOCHS=30
MODEL="ViT-B-16"
#MODEL="ViT-B-16-512"
#MODEL="ViT-L-16"
TRAIN_DATA="/lpai/dataset/cc12m/0-1-0/cc12m-wds/cc12m-train-{0000..2175}.tar"
PROJECT_NAME="test"
ALPHA=1
BETA=0.5


# CUDA_VISIBLE_DEVICES=7 python -m main.run --logs=$OUTPATH \
torchrun --nnodes 1 --nproc_per_node 8 -m main.run \
    --save-frequency 2 --report-to wandb \
    --wandb-project-name=$PROJECT_NAME \
    --train-data=$TRAIN_DATA --train-num-samples 10968539 --warmup 10000  \
    --batch-size=$BS --lr=1e-3 --wd=0.1 --epochs=30 --workers=2 --model "ViT-B-16" \
    --precision amp --dataset-type webdataset \
    --clip-inModality-loss --clip-loss \
    --alpha=1 --beta=0.5 --nl_semantic_supervision \
    --separate_text --separate_image 