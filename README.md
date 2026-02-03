
# Running the Resnet-50 Training, Saving the model and calculating the Conformal Prediction at alpha =0.1
python ./training_script_resnet_test_run_7.py

# Running the Vision Transformer Training, Saving the model and calculating the Conformal Prediction at alpha =0.1
python -u ./training_script_vision_transformer_test_run_7.py

# Calculation of alpha values, loading of saved models


## Calculation of alpha values for Resnet-50
python conformal_multi_alpha_resnet.py \
    --test_run 1 \
    --resnet_ckpt_path "/...../...../best_model_resnet_modified_300_epochs_...pth"

## Calculation of alpha values for Resnet-50
python conformal_multi_alpha_Vit.py \
 --test_run 1 \
 --vit_ckpt_path "/path/to/Pretrained_vit_scratch_training_test_run_1_save_after_200_epochs.pth" \
 --data_dir "/leonardo_work/IscrC_SKIDD-AI/alitariqnagi_work/training_data_split_run_1"
