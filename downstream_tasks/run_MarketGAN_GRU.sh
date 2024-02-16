python run_prediction.py \
--task MarketGAN \
--epoch 150 \
--seed 42 \
--exp_name one_step_prediction_GRU_150 \
--device cuda:3 \
--batch_size 512 \
--prediction_model GRU