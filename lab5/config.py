"""All lab settings."""
from pathlib import Path

# Paths
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "output"                 # all results
CACHE = OUT / "cache"                 # saved predictions, so they are not recomputed
DATASETS = ROOT / "datasets"
DATA_DIR = ROOT / "data"
IMAGES_DIR = DATA_DIR / "images"      # photos for task 2 (*.jpg, *.png)
VIDEOS_DIR = DATA_DIR / "videos"      # videos for task 3, the first one alphabetically is used
WEIGHTS_DIR = ROOT / "weights"        # model weights, downloaded automatically

# Models
MODELS = {"nano": "yolov8n.pt", "small": "yolov8s.pt"}
MAIN_MODEL = "nano"                   # model for tasks 1-5 and the error analysis

.
DEVICE = None
_DEV = {} if DEVICE is None else {"device": DEVICE}

PRED = dict(conf=0.25, iou=0.7, max_det=300, **_DEV)   # regular inference
EVAL = dict(conf=0.001, iou=0.7, max_det=300, **_DEV)  # for metrics, same as model.val
VAL = dict(plots=False, verbose=False, **_DEV)          # for model.val

# Data for metrics
# Main dataset: a subset of COCO val2017 (created by the "dataset" step).
COCO_SUBSET_DIR = DATASETS / "coco_val2017_subset"
COCO_SUBSET_YAML = COCO_SUBSET_DIR / "data.yaml"
COCO_SUBSET_SIZE = 500
FALLBACK_DATA = "coco128.yaml"

# Study parameters
CONF_THRESHOLDS = [0.1, 0.25, 0.5, 0.75, 0.9]        
CONF_GRID = [round(0.05 * i, 2) for i in range(1, 20)]  # 0.05..0.95, used to find the optimal c
IMG_SIZES = [320, 480, 640, 800, 960]
IOU_MATCH = 0.5                       # a detection is a TP if IoU with ground truth >= 0.5
NMS_IOU_THRESHOLDS = [0.3, 0.5, 0.7, 0.9]

VIDEO_SOURCE = None
VIDEO_MAX_FRAMES = 300

FPS_N_IMAGES = 50                     # images used to measure FPS
FPS_REPEATS = 3                       # repeats, the median is reported
REALTIME_FPS = 25                     # threshold for choosing the optimal configuration
MIN_INSTANCES = 5                     # classes with fewer objects are not ranked (AP is noise)
SEED = 0
