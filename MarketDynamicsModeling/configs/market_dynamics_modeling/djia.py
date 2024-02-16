market_dynamics_model = dict(
    data_path='../Market_GAN_AAAI/data/DJI/DJI_data.csv',
    filter_strength=1,
    slope_interval=[-0.01, 0.01],
    dynamic_number=3,
    max_length_expectation=140,
    OE_BTC=False,
    PM='',
    process_datafile_path=
    '',
    market_dynamic_labeling_visualization_paths=[
    ],
    key_indicator='Adj Close',
    timestamp='Date',
    tic='tic',
    labeling_method='quantile',
    min_length_limit=50,
    merging_metric='DTW_distance',
    merging_threshold=0.03,
    merging_dynamic_constraint=1,
    exp_name='DJI_50',
    type='Linear_Market_Dynamics_Model',
    market_dynamic_modeling_visualization_paths=[
    ],
    market_dynamic_modeling_analysis_paths=[
    ])
task_name = 'custom'
dataset_name = 'custom'
