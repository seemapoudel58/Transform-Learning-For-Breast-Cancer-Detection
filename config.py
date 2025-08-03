import os
import torch

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
MODEL_SAVE_PATH = os.path.join(PROJECT_ROOT, 'models')
LOG_DIR = os.path.join(PROJECT_ROOT, 'logs')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(DEVICE)

IMAGE_SIZE = (224, 224)
IMAGE_MEAN = [0.485, 0.456, 0.406]  
IMAGE_STD = [0.229, 0.224, 0.225] 

SPLITS = ['train', 'valid', 'test']
CLASSES = ['0', '1'] 

MODEL_NAME = 'resnet50'
BATCH_SIZE = 8
EPOCHS = 5
LEARNING_RATE = 1e-4
SEED = 42
NUM_WORKERS = 2

def get_image_folder(split, cls):
    return os.path.join(DATA_DIR, split, cls)
