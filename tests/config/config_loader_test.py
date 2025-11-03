"""
Tests for configuration loading and parsing.
"""

import tempfile
from pathlib import Path

import pytest

from config.config import (
    Augmentation,
    AugmentationStep,
    Config,
    Embedding,
    InputDirs,
    Model,
    Optuna,
    OptunaRanges,
    OutputDirs,
    Pipeline,
)


class TestAugmentationStep:
    """Tests for AugmentationStep dataclass."""
    
    def test_from_dict_gaussian_noise(self):
        """Test parsing gaussian noise augmentation step."""
        data = {
            'type': 'gaussian_noise',
            'noise_factor': 0.001,
            'p': 1.0
        }
        step = AugmentationStep.from_dict(data)
        
        assert step.type == 'gaussian_noise'
        assert step.noise_factor == 0.001
        assert step.p == 1.0
        assert step.scale_range is None
        assert step.max_mask_percentage is None
    
    def test_from_dict_random_scaling(self):
        """Test parsing random scaling augmentation step."""
        data = {
            'type': 'random_scaling',
            'scale_range': [0.95, 1.05]
        }
        step = AugmentationStep.from_dict(data)
        
        assert step.type == 'random_scaling'
        assert step.scale_range == [0.95, 1.05]
        assert step.noise_factor is None
        assert step.p is None
    
    def test_from_dict_time_masking(self):
        """Test parsing time masking augmentation step."""
        data = {
            'type': 'time_masking',
            'max_mask_percentage': 0.01
        }
        step = AugmentationStep.from_dict(data)
        
        assert step.type == 'time_masking'
        assert step.max_mask_percentage == 0.01


class TestAugmentation:
    """Tests for Augmentation dataclass."""
    
    def test_from_dict(self):
        """Test parsing augmentation configuration."""
        data = {
            'AUGMENT': False,
            'REPEATS': 100,
            'AUG_FILE': 'data/augmented_data.npz',
            'SEED': 42,
            'STEPS': [
                {'type': 'gaussian_noise', 'noise_factor': 0.001, 'p': 1.0},
                {'type': 'random_scaling', 'scale_range': [0.95, 1.05]}
            ]
        }
        aug = Augmentation.from_dict(data)
        
        assert aug.enabled is False
        assert aug.repeats == 100
        assert aug.aug_dir == Path('data/augmented_data.npz')
        assert aug.seed == 42
        assert len(aug.steps) == 2
        assert aug.steps[0].type == 'gaussian_noise'
        assert aug.steps[1].type == 'random_scaling'


class TestEmbedding:
    """Tests for Embedding dataclass."""
    
    def test_from_dict(self):
        """Test parsing embedding configuration."""
        data = {
            'MODEL': 'nomic-embed-text-v1.5',
            'DIMENSION': 64,
            'PRE_PRECISION': 2,
            'OUTPUT_FILE': 'data/processed_data/'
        }
        emb = Embedding.from_dict(data)
        
        assert emb.model == 'nomic-embed-text-v1.5'
        assert emb.dimension == 64
        assert emb.pre_precision == 2
        assert emb.output_dir == Path('data/processed_data/')


class TestPipeline:
    """Tests for Pipeline dataclass."""
    
    def test_from_dict(self):
        """Test parsing pipeline configuration."""
        data = {
            'TRAIN': True,
            'TEST': False
        }
        pipeline = Pipeline.from_dict(data)
        
        assert pipeline.train is True
        assert pipeline.test is False


class TestInputDirs:
    """Tests for InputDirs dataclass."""
    
    def test_from_dict(self):
        """Test parsing input directories configuration."""
        data = {
            'TRAIN_FILE': 'data/ae.train',
            'TEST_FILE': 'data/ae.test'
        }
        dirs = InputDirs.from_dict(data)
        
        assert dirs.train_file_dir == Path('data/ae.train')
        assert dirs.test_file_dir == Path('data/ae.test')


class TestOutputDirs:
    """Tests for OutputDirs dataclass."""
    
    def test_from_dict(self):
        """Test parsing output directories configuration."""
        data = {
            'PROCESSED': 'data/processed_data'
        }
        dirs = OutputDirs.from_dict(data)
        
        assert dirs.processed_file_dir == Path('data/processed_data')


class TestOptunaRanges:
    """Tests for OptunaRanges dataclass."""
    
    def test_from_dict(self):
        """Test parsing Optuna ranges configuration."""
        data = {
            'LEARNING_RATE': [1e-5, 1e-2],
            'DROPOUT': [0.1, 0.5],
            'CONV_CHANNELS': [64, 128, 256],
            'HIDDEN_DIM': [32, 64, 128],
            'KERNEL_SIZE': [3, 5, 7],
            'BATCH_SIZE': [16, 32, 64]
        }
        ranges = OptunaRanges.from_dict(data)
        
        assert ranges.learning_rate == [1e-5, 1e-2]
        assert ranges.dropout == [0.1, 0.5]
        assert ranges.conv_channels == [64, 128, 256]
        assert ranges.hidden_dim == [32, 64, 128]
        assert ranges.kernel_size == [3, 5, 7]
        assert ranges.batch_size == [16, 32, 64]


class TestOptuna:
    """Tests for Optuna dataclass."""
    
    def test_from_dict(self):
        """Test parsing Optuna configuration."""
        data = {
            'ENABLED': False,
            'N_TRIALS': 50,
            'STUDY_NAME': 'HAIKU_speaker_recognition',
            'FIGURES_DIR': 'reports/optuna/figures',
            'STUDY_DIR': 'reports/optuna/study',
            'BEST_CONFIG_DIR': 'reports/optuna/best_config',
            'RANGES': {
                'LEARNING_RATE': [1e-5, 1e-2],
                'DROPOUT': [0.1, 0.5],
                'CONV_CHANNELS': [64, 128, 256],
                'HIDDEN_DIM': [32, 64, 128],
                'KERNEL_SIZE': [3, 5, 7],
                'BATCH_SIZE': [16, 32, 64]
            }
        }
        optuna = Optuna.from_dict(data)
        
        assert optuna.enabled is False
        assert optuna.n_trials == 50
        assert optuna.study_name == 'HAIKU_speaker_recognition'
        assert optuna.figures_dir == Path('reports/optuna/figures')
        assert optuna.study_dir == Path('reports/optuna/study')
        assert optuna.best_config_dir == Path('reports/optuna/best_config')
        assert isinstance(optuna.ranges, OptunaRanges)


class TestModel:
    """Tests for Model dataclass."""
    
    def test_from_dict(self):
        """Test parsing model configuration."""
        data = {
            'LOAD_BEST_CONFIG': True,
            'NUM_CLASSES': 9,
            'EMBEDDING_DIM': 64,
            'KERNEL_SIZE': 3,
            'CONV_CHANNELS': 128,
            'DROPOUT': 0.3,
            'INPUT_CHANNELS': 12,
            'HIDDEN_DIM': 64,
            'LEARNING_RATE': 0.007,
            'BATCH_SIZE': 32,
            'NUM_EPOCHS': 10,
            'K_FOLDS': 2,
            'DEVICE': 'auto'
        }
        model = Model.from_dict(data)
        
        assert model.load_best_config is True
        assert model.num_classes == 9
        assert model.embedding_dim == 64
        assert model.kernel_size == 3
        assert model.conv_channels == 128
        assert model.dropout == 0.3
        assert model.input_channels == 12
        assert model.hidden_dim == 64
        assert model.learning_rate == 0.007
        assert model.batch_size == 32
        assert model.num_epochs == 10
        assert model.k_folds == 2
        assert model.device == 'auto'
    
    def test_device_literal_values(self):
        """Test that device accepts only valid literal values."""
        for device in ['cuda', 'cpu', 'auto']:
            data = {
                'LOAD_BEST_CONFIG': True,
                'NUM_CLASSES': 9,
                'EMBEDDING_DIM': 64,
                'KERNEL_SIZE': 3,
                'CONV_CHANNELS': 128,
                'DROPOUT': 0.3,
                'INPUT_CHANNELS': 12,
                'HIDDEN_DIM': 64,
                'LEARNING_RATE': 0.007,
                'BATCH_SIZE': 32,
                'NUM_EPOCHS': 10,
                'K_FOLDS': 2,
                'DEVICE': device
            }
            model = Model.from_dict(data)
            assert model.device == device


class TestConfig:
    """Tests for Config dataclass."""
    
    def test_from_dict_full_config(self):
        """Test parsing full configuration from dictionary."""
        data = {
            'DATA_URL': 'https://archive.ics.uci.edu/ml/machine-learning-databases/JapaneseVowels/',
            'SEED': 42,
            'MAX_LEN': 29,
            'N_FEATURES': 12,
            'PIPELINE': {
                'TRAIN': True,
                'TEST': False
            },
            'AUGMENTATION': {
                'AUGMENT': False,
                'REPEATS': 100,
                'AUG_FILE': 'data/augmented_data.npz',
                'SEED': 42,
                'STEPS': [
                    {'type': 'gaussian_noise', 'noise_factor': 0.001, 'p': 1.0}
                ]
            },
            'EMBEDDING': {
                'MODEL': 'nomic-embed-text-v1.5',
                'DIMENSION': 64,
                'PRE_PRECISION': 2,
                'OUTPUT_FILE': 'data/processed_data/'
            },
            'INPUT_DIRS': {
                'TRAIN_FILE': 'data/ae.train',
                'TEST_FILE': 'data/ae.test'
            },
            'OUTPUT_DIRS': {
                'PROCESSED': 'data/processed_data'
            },
            'MODEL': {
                'LOAD_BEST_CONFIG': True,
                'NUM_CLASSES': 9,
                'EMBEDDING_DIM': 64,
                'KERNEL_SIZE': 3,
                'CONV_CHANNELS': 128,
                'DROPOUT': 0.3,
                'INPUT_CHANNELS': 12,
                'HIDDEN_DIM': 64,
                'LEARNING_RATE': 0.007,
                'BATCH_SIZE': 32,
                'NUM_EPOCHS': 10,
                'K_FOLDS': 2,
                'DEVICE': 'auto'
            },
            'OPTUNA': {
                'ENABLED': False,
                'N_TRIALS': 50,
                'STUDY_NAME': 'HAIKU_speaker_recognition',
                'FIGURES_DIR': 'reports/optuna/figures',
                'STUDY_DIR': 'reports/optuna/study',
                'BEST_CONFIG_DIR': 'reports/optuna/best_config',
                'RANGES': {
                    'LEARNING_RATE': [1e-5, 1e-2],
                    'DROPOUT': [0.1, 0.5],
                    'CONV_CHANNELS': [64, 128, 256],
                    'HIDDEN_DIM': [32, 64, 128],
                    'KERNEL_SIZE': [3, 5, 7],
                    'BATCH_SIZE': [16, 32, 64]
                }
            }
        }
        
        config = Config.from_dict(data)
        
        assert config.data_url == 'https://archive.ics.uci.edu/ml/machine-learning-databases/JapaneseVowels/'
        assert config.seed == 42
        assert config.max_length == 29
        assert config.n_features == 12
        assert isinstance(config.pipeline, Pipeline)
        assert isinstance(config.augmentation, Augmentation)
        assert isinstance(config.embedding, Embedding)
        assert isinstance(config.input_dirs, InputDirs)
        assert isinstance(config.output_dirs, OutputDirs)
        assert isinstance(config.model, Model)
        assert isinstance(config.optuna, Optuna)
    
    def test_from_yaml(self):
        """Test loading configuration from YAML file."""
        yaml_content = """
DATA_URL: "https://archive.ics.uci.edu/ml/machine-learning-databases/JapaneseVowels/"
SEED: 42
MAX_LEN: 29
N_FEATURES: 12

PIPELINE:
  TRAIN: true
  TEST: false

AUGMENTATION:
  AUGMENT: false
  REPEATS: 100
  AUG_FILE: "data/augmented_data.npz"
  SEED: 42
  STEPS:
    - type: gaussian_noise
      noise_factor: 0.001
      p: 1.0

EMBEDDING:
  MODEL: "nomic-embed-text-v1.5"
  DIMENSION: 64
  PRE_PRECISION: 2
  OUTPUT_FILE: "data/processed_data/"

INPUT_DIRS:
  TRAIN_FILE: "data/ae.train"
  TEST_FILE: "data/ae.test"

OUTPUT_DIRS:
  PROCESSED: "data/processed_data"

MODEL:
  LOAD_BEST_CONFIG: true
  NUM_CLASSES: 9
  EMBEDDING_DIM: 64
  KERNEL_SIZE: 3
  CONV_CHANNELS: 128
  DROPOUT: 0.3
  INPUT_CHANNELS: 12
  HIDDEN_DIM: 64
  LEARNING_RATE: 0.007
  BATCH_SIZE: 32
  NUM_EPOCHS: 10
  K_FOLDS: 2
  DEVICE: "auto"

OPTUNA:
  ENABLED: false
  N_TRIALS: 50
  STUDY_NAME: "HAIKU_speaker_recognition"
  FIGURES_DIR: "reports/optuna/figures"
  STUDY_DIR: "reports/optuna/study"
  BEST_CONFIG_DIR: "reports/optuna/best_config"
  RANGES:
    LEARNING_RATE: [1e-5, 1e-2]
    DROPOUT: [0.1, 0.5]
    CONV_CHANNELS: [64, 128, 256]
    HIDDEN_DIM: [32, 64, 128]
    KERNEL_SIZE: [3, 5, 7]
    BATCH_SIZE: [16, 32, 64]
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            temp_path = f.name
        
        try:
            config = Config.from_yaml(temp_path)
            
            assert config.data_url == 'https://archive.ics.uci.edu/ml/machine-learning-databases/JapaneseVowels/'
            assert config.seed == 42
            assert config.model.num_classes == 9
            assert config.model.learning_rate == 0.007
            assert config.augmentation.enabled is False
            assert len(config.augmentation.steps) == 1
        finally:
            Path(temp_path).unlink()
    
    def test_from_yaml_actual_config_file(self):
        """Test loading from the actual config.yaml file if it exists."""
        config_path = Path("config/config.yaml")
        
        if not config_path.exists():
            pytest.skip("config/config.yaml not found")
        
        config = Config.from_yaml(config_path)
        
        # Verify key values from the actual config
        assert config.seed == 42
        assert config.max_length == 29
        assert config.n_features == 12
        assert config.model.num_classes == 9
        assert config.model.embedding_dim == 64
        assert config.optuna.study_name == "HAIKU_speaker_recognition"
    
    def test_path_objects_created(self):
        """Test that Path objects are properly created."""
        data = {
            'DATA_URL': 'https://example.com',
            'SEED': 42,
            'MAX_LEN': 29,
            'N_FEATURES': 12,
            'PIPELINE': {'TRAIN': True, 'TEST': False},
            'AUGMENTATION': {
                'AUGMENT': False,
                'REPEATS': 100,
                'AUG_FILE': 'data/augmented_data.npz',
                'SEED': 42,
                'STEPS': []
            },
            'EMBEDDING': {
                'MODEL': 'test-model',
                'DIMENSION': 64,
                'PRE_PRECISION': 2,
                'OUTPUT_FILE': 'data/processed_data/'
            },
            'INPUT_DIRS': {
                'TRAIN_FILE': 'data/ae.train',
                'TEST_FILE': 'data/ae.test'
            },
            'OUTPUT_DIRS': {
                'PROCESSED': 'data/processed_data'
            },
            'MODEL': {
                'LOAD_BEST_CONFIG': True,
                'NUM_CLASSES': 9,
                'EMBEDDING_DIM': 64,
                'KERNEL_SIZE': 3,
                'CONV_CHANNELS': 128,
                'DROPOUT': 0.3,
                'INPUT_CHANNELS': 12,
                'HIDDEN_DIM': 64,
                'LEARNING_RATE': 0.007,
                'BATCH_SIZE': 32,
                'NUM_EPOCHS': 10,
                'K_FOLDS': 2,
                'DEVICE': 'auto'
            },
            'OPTUNA': {
                'ENABLED': False,
                'N_TRIALS': 50,
                'STUDY_NAME': 'test',
                'FIGURES_DIR': 'reports/figures',
                'STUDY_DIR': 'reports/study',
                'BEST_CONFIG_DIR': 'reports/config',
                'RANGES': {
                    'LEARNING_RATE': [1e-5, 1e-2],
                    'DROPOUT': [0.1, 0.5],
                    'CONV_CHANNELS': [64, 128],
                    'HIDDEN_DIM': [32, 64],
                    'KERNEL_SIZE': [3, 5],
                    'BATCH_SIZE': [16, 32]
                }
            }
        }
        
        config = Config.from_dict(data)
        
        assert isinstance(config.augmentation.aug_dir, Path)
        assert isinstance(config.embedding.output_dir, Path)
        assert isinstance(config.input_dirs.train_file_dir, Path)
        assert isinstance(config.input_dirs.test_file_dir, Path)
        assert isinstance(config.output_dirs.processed_file_dir, Path)
        assert isinstance(config.optuna.figures_dir, Path)
        assert isinstance(config.optuna.study_dir, Path)
        assert isinstance(config.optuna.best_config_dir, Path)