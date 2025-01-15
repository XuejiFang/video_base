# 0. Set up the environment
export MODEL_ID="cogvideox-2b"
export MODEL="outputs/$MODEL_ID"
export SAVE_DIR="outputs_eval/vbench/$MODEL_ID"
export OUTPIT_DIR="results/vbench/$MODEL_ID"
# 1. Generating Videos
# bash scripts/train/eval_sample.sh infer_vbench.py $MODEL

# 2. Evaluating Videos
# echo "[1/16] evaluating dimension: aesthetic_quality"
# vbench evaluate --ngpus 8 --output_path $OUTPIT_DIR --dimension aesthetic_quality --videos_path $SAVE_DIR/overall_consistency
# wait

# echo "[2/16] evaluating dimension: appearance_style"
# vbench evaluate --ngpus 8 --output_path $OUTPIT_DIR --dimension appearance_style --videos_path $SAVE_DIR/appearance_style
# wait

# echo "[3/16] evaluating dimension: background_consistency"
# vbench evaluate --ngpus 8 --output_path $OUTPIT_DIR --dimension background_consistency --videos_path $SAVE_DIR/scene
# wait

# echo "[4/16] evaluating dimension: color"
# vbench evaluate --ngpus 8 --output_path $OUTPIT_DIR --dimension color --videos_path $SAVE_DIR/color
# wait

# echo "[5/16] evaluating dimension: dynamic_degree"
# vbench evaluate --ngpus 8 --output_path $OUTPIT_DIR --dimension dynamic_degree --videos_path $SAVE_DIR/subject_consistency
# wait

# echo "[6/16] evaluating dimension: subject_consistency"
# vbench evaluate --ngpus 8 --output_path $OUTPIT_DIR --dimension subject_consistency --videos_path $SAVE_DIR/subject_consistency
# wait

# echo "[7/16] evaluating dimension: motion_smoothness"
# vbench evaluate --ngpus 8 --output_path $OUTPIT_DIR --dimension motion_smoothness --videos_path $SAVE_DIR/subject_consistency
# wait

# echo "[8/16] evaluating dimension: human_action"
# vbench evaluate --ngpus 8 --output_path $OUTPIT_DIR --dimension human_action --videos_path $SAVE_DIR/human_action
# wait

# echo "[9/16] evaluating dimension: imaging_quality"
# vbench evaluate --ngpus 8 --output_path $OUTPIT_DIR --dimension imaging_quality --videos_path $SAVE_DIR/overall_consistency
# wait

# echo "[10/16] evaluating dimension: multiple_objects"
# vbench evaluate --ngpus 8 --output_path $OUTPIT_DIR --dimension multiple_objects --videos_path $SAVE_DIR/multiple_objects
# wait

# echo "[11/16] evaluating dimension: object_class"
# vbench evaluate --ngpus 8 --output_path $OUTPIT_DIR --dimension object_class --videos_path $SAVE_DIR/object_class
# wait

# echo "[12/16] evaluating dimension: overall_consistency"
# vbench evaluate --ngpus 8 --output_path $OUTPIT_DIR --dimension overall_consistency --videos_path $SAVE_DIR/overall_consistency
# wait

# echo "[13/16] evaluating dimension: scene"
# vbench evaluate --ngpus 8 --output_path $OUTPIT_DIR --dimension scene --videos_path $SAVE_DIR/scene
# wait

# echo "[14/16] evaluating dimension: spatial_relationship"
# vbench evaluate --ngpus 8 --output_path $OUTPIT_DIR --dimension spatial_relationship --videos_path $SAVE_DIR/spatial_relationship
# wait

echo "[15/16] evaluating dimension: temporal_flickering"
vbench evaluate --ngpus 8 --output_path $OUTPIT_DIR --dimension temporal_flickering --videos_path $SAVE_DIR/temporal_flickering
wait    

echo "[16/16] evaluating dimension: temporal_style"
vbench evaluate --ngpus 8 --output_path $OUTPIT_DIR --dimension temporal_style --videos_path $SAVE_DIR/temporal_style
wait
