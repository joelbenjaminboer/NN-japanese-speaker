# japanese speaker recognition
This project develops a machine learning classifier to identify one of nine japanese male speakers based on short 12-channel time-series recordings of the vowel /ae/.

## Project Organization

```
├── config
│   ├── config.py
│   └── config.yaml
├── data
│   ├── ae.test
│   ├── ae.train
│   └── processed_data
├── docs
│   └── Project_JapVowels.pdf
├── habrok_outputs
├── habrok_shared
│   └── optuna_study.db
├── IMPORTANT OUTPUTS
├── japanese_speaker_recognition
│   ├── core
│   │   ├── base.py
│   │   └── registry.py
│   ├── data_augmentation.py
│   ├── data_embedding.py
│   ├── dataset.py
│   ├── features
│   │   └── flatten.py
│   ├── __init__.py
│   ├── metrics
│   │   ├── classification.py
│   │   └── __init__.py
│   ├── modeling
│   │   ├── init.py
│   │   ├── predict.py
│   │   └── train.py
│   ├── models
│   │   ├── HAIKU.py
│   │   ├── __pycache__
│   │   │   ├── cnn.cpython-311.pyc
│   │   │   └── HAIKU.cpython-311.pyc
│   │   └── random_forest.py
│   ├── optimization
│   │   ├── __init__.py
│   │   └── optuna_tuner.py
│   ├── plots.py
├── main.py
├── main.sh
├── Makefile
├── models
├── notebooks
├── optuna_study.db
├── optuna_study_v3.db
├── parallel_tuning.py
├── pyproject.toml
├── README.md
├── references
├── reports
│   ├── figures
│   │   ├── frame_level_correlation.png
│   │   ├── frame_level_correlation_train.png
│   │   ├── lpc_correlation_heatmap.png
│   │   ├── training_10K.png
│   │   └── training_history.png
│   └── optuna
│       ├── best_config
│       │   └── best_model_config.yaml
│       ├── figures
│       │   ├── optuna_optimization_history.png
│       │   ├── optuna_parallel_coordinate.png
│       │   └── optuna_param_importances.png
│       └── study
│           └── optuna_study.pkl
├── requirements.txt
├── tests
│   └── config
│       └── config_loader_test.py
├── train_haiku.sh
├── utils
│   ├── __pycache__
│   │   └── utils.cpython-311.pyc
│   └── utils.py
└── uv.lock
```

--------

# About the project

This project aims to further research the classification strategy of Kaur et al. (2025)doi:(10.18653/v1/2025.acl-long.1557)
With this strategy we use the Nomic text embedding model to embed our timeseries data into a vector. This combined with the original data gives the data more dimension which in turn can be fed to our fairly simple HAIKU model to predict better then most complex LLM solutions to this problem.

Our goal is to explore more compact models and find a model that has a better accuracy with less parameters then other known solutions.

# How to run the code

1. All our variables and features can be used from the config file inside of config/config.yaml. 

2. Everything can be tweaked to run the model in the desired format. finally to start the process 

3. Run python main.py

note: When making tweaks to the dataset make sure to give it a new unique key inside of "EMBEDDING"