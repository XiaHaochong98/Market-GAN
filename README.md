# Market-GAN
Implmentation of [Market-GAN: Adding Control to Financial Market Data Generation with Semantic Context (AAAI24)](https://arxiv.org/abs/2309.07708)
You may download the [model checkpoint](https://entuedu-my.sharepoint.com/:f:/g/personal/haochong001_e_ntu_edu_sg/ElopwODLW9xAuaWEAF7y0AcBogQMQexrrTbioUspuKAs5Q?e=t7IXDc) and use it by putting the files into the output/ folder.
# Installation
Use the AAAI_MarketGAN.yml to create a conda environment
``` conda env create -f AAAI_MarketGAN.yml ```

# Usage
This repo contains (1) The Market Dynamics Modeling tool, (2) Market-GAN, (3) Example usage of generated data on the downstream task.
Follow these steps to walkthrogh the whole process.

# I) Market Dynamics Modeling
**In the MarketDynamicsModeling folder**
## 1. Download DJI dataset from Yahoo Finance
``` python get_DJI_dataset.py ```
## 2. Run market dynamics modeling 
(1) Move the DJI_data.csv to the data folder of Market_GAN_AAAI/data  

(2) Change the "process_datafile_path=" in the config file MarketDynamicsModeling/configs/market_dynamics_modeling/djia.py to the path of the data set you downloaded

(3) Run the following command to label the data set.
(the file is at configs/market_dynamics_modeling/djia.py)

use ``` python tools/market_dynamics_labeling/run.py``` ( we have already done the (1) and (2) for you)

or 

use ``` python run.py --config {path of you djia.py config file} ``` if you want to implement (1) and (2) by yourself

# II) Train Market-GAN
**In the Market_GAN_AAAI folder**
## 1. Pre-train condition supervisors
``` sh Pretrain_DJI_V2_50.sh ```
## 2. Train Market-GAN
``` sh DJI_V2_RT_train.sh ```
## 3. Evaluate Market-GAN
``` sh Evaluate_DJI_V2_RT.sh ```
## 4. Visualize the results
``` sh Plot_DJI_V2_RT.sh ```

The results will be saved to the Market_GAN_AAAI/output.
You may download the [model checkpoint](https://entuedu-my.sharepoint.com/:f:/g/personal/haochong001_e_ntu_edu_sg/ElopwODLW9xAuaWEAF7y0AcBogQMQexrrTbioUspuKAs5Q?e=t7IXDc) and use it by putting the files into the output/ folder.

# III) Use Market-GAN to generate data for the downstream task
## 1.Generate data for the downstream tasks
(1) Export the model information for inference by running ``` sh DJI_V2_RT_info_export.sh ```
(2) Generate the data by running ``` sh run_MarketGAN.sh ``` in the /service folder
(3) Move the generated data to downstream_tasks/data/downstream_tasks/data
(4) Run scripts for different model like ``` sh run_MarketGAN.sh ``` in downstream_tasks/
## 2.Prepare orginal data for the downstream tasks
(1) Run ``` sh DJI.sh ```  in downstream_tasks/data/
(2) Run scripts for different model like ``` sh run_real.sh ``` in downstream_tasks/

# Reference
We thank these projects in helping the development of Market-GAN:
[Codebase for "Time-series Generative Adversarial Networks (TimeGAN)"](https://github.com/jsyoon0823/TimeGAN?tab=readme-ov-file#codebase-for-time-series-generative-adversarial-networks-timegan) \
[timegan-pytorch](https://github.com/birdx0810/timegan-pytorch) \
[Time Series Library (TSlib)](https://github.com/thuml/Time-Series-Library)
