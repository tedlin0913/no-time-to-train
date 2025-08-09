#!/bin/bash

# Melon dataset pipeline
DATASET_NAME=melon
DATASET_PATH=data/melon
CAT_NAMES=flower,leaf
CATEGORY_NUM=2
SHOT=1
YAML_PATH=no_time_to_train/pl_configs/matching_melon_dataset.yaml
PATH_TO_SAVE_CKPTS=./tmp_ckpts/melon
mkdir -p $PATH_TO_SAVE_CKPTS

echo "Starting melon dataset pipeline..."

# STEP 1: Fill memory with references
echo "Step 1: Filling memory with references..."
python run_lightening.py test --config $YAML_PATH \
    --model.test_mode fill_memory \
    --out_path $PATH_TO_SAVE_CKPTS/$DATASET_NAME\_$SHOT\_refs_memory.pth \
    --model.init_args.dataset_cfgs.fill_memory.root $DATASET_PATH/images \
    --model.init_args.dataset_cfgs.fill_memory.json_file $DATASET_PATH/annotations/custom_references_with_segm.json \
    --model.init_args.dataset_cfgs.fill_memory.memory_pkl $DATASET_PATH/annotations/custom_references_with_segm.pkl \
    --model.init_args.dataset_cfgs.fill_memory.memory_length $SHOT \
    --model.init_args.dataset_cfgs.fill_memory.cat_names $CAT_NAMES \
    --model.init_args.model_cfg.dataset_name $DATASET_NAME \
    --model.init_args.model_cfg.memory_bank_cfg.length $SHOT \
    --model.init_args.model_cfg.memory_bank_cfg.category_num $CATEGORY_NUM \
    --trainer.devices 1

echo "Step 1 completed: Memory filled"

# STEP 2: Skip post-processing for small dataset
echo "Step 2: post-processing"

python run_lightening.py test --config $YAML_PATH \
    --model.test_mode postprocess_memory \
    --ckpt_path $PATH_TO_SAVE_CKPTS/$DATASET_NAME\_$SHOT\_refs_memory.pth \
    --out_path $PATH_TO_SAVE_CKPTS/$DATASET_NAME\_$SHOT\_refs_memory_postprocessed.pth \
    --model.init_args.model_cfg.dataset_name $DATASET_NAME \
    --model.init_args.model_cfg.memory_bank_cfg.length $SHOT \
    --model.init_args.model_cfg.memory_bank_cfg.category_num $CATEGORY_NUM \
    --trainer.devices 1

echo "Step 3: Inference"

ONLINE_VIS=True
VIS_THR=0.4
python run_lightening.py test --config $YAML_PATH \
    --model.test_mode test \
    --ckpt_path $PATH_TO_SAVE_CKPTS/$DATASET_NAME\_$SHOT\_refs_memory_postprocessed.pth \
    --model.init_args.model_cfg.dataset_name $DATASET_NAME \
    --model.init_args.model_cfg.memory_bank_cfg.length $SHOT \
    --model.init_args.model_cfg.memory_bank_cfg.category_num $CATEGORY_NUM \
    --model.init_args.model_cfg.test.imgs_path $DATASET_PATH/images \
    --model.init_args.model_cfg.test.online_vis $ONLINE_VIS \
    --model.init_args.model_cfg.test.vis_thr $VIS_THR \
    --model.init_args.dataset_cfgs.test.root $DATASET_PATH/images \
    --model.init_args.dataset_cfgs.test.json_file $DATASET_PATH/annotations/custom_targets.json \
    --model.init_args.dataset_cfgs.test.cat_names $CAT_NAMES \
    --trainer.devices 1

echo "Pipeline completed successfully!"

