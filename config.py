import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent

YOLO_MODEL = os.getenv("YOLO_MODEL", "yolov8x.pt")
FURNITURE_CONF = 0.12
FURNITURE_IMGSZ = 1024

FURNITURE_CLASS_CONF = {
    "bed": 0.35,
    "chair": 0.35,
    "sofa": 0.35,
    "couch": 0.14,
    "dining table": 0.35,
    "bench": 0.35,
    "tv": 0.35,
}

FURNITURE_CLASSES = {
    "bed": "bed",
    "chair": "chair",
    "couch": "sofa",
    "sofa": "sofa",
    "dining table": "table",
    "bench": "sofa",
    "tv": "table",
}

# HORIZONNET_DIR = Path(os.getenv("HORIZONNET_DIR", str(ROOT_DIR.parent / "HorizonNet")))
# HORIZONNET_WEIGHTS = Path(os.getenv("HORIZONNET_WEIGHTS", str(HORIZONNET_DIR / "model.pth")))
HORIZONNET_DIR = Path(__file__).resolve().parent
HORIZONNET_WEIGHTS = ROOT_DIR / "model.pt"

ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY", "OgUBvraBD702WWqYgfoC")
ROBOFLOW_URL = os.getenv("ROBOFLOW_URL", "https://detect.roboflow.com/door-window-detection-pipvh/1")

RF_CONNECT_TIMEOUT = int(os.getenv("RF_CONNECT_TIMEOUT", "20"))
RF_READ_TIMEOUT = int(os.getenv("RF_READ_TIMEOUT", "60"))
RF_RETRY_TOTAL = int(os.getenv("RF_RETRY_TOTAL", "3"))
RF_RETRY_CONNECT = int(os.getenv("RF_RETRY_CONNECT", "3"))
RF_RETRY_READ = int(os.getenv("RF_RETRY_READ", "3"))
RF_RETRY_BACKOFF = float(os.getenv("RF_RETRY_BACKOFF", "1.0"))
RF_FAIL_SOFT = os.getenv("RF_FAIL_SOFT", "1") == "1"

OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", str(ROOT_DIR / "outputs")))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ROOM_SCALE = float(os.getenv("ROOM_SCALE", "5.0"))
WALL_THICKNESS = float(os.getenv("WALL_THICKNESS", "0.18"))
DOOR_WIDTH_M = float(os.getenv("DOOR_WIDTH_M", "0.90"))
WINDOW_WIDTH_M = float(os.getenv("WINDOW_WIDTH_M", "1.20"))
DXF_SCALE = float(os.getenv("DXF_SCALE", "100"))

PEAK_THRESH = float(os.getenv("PEAK_THRESH", "0.38"))
MIN_PEAK_DISTANCE = int(os.getenv("MIN_PEAK_DISTANCE", "28"))
CLUSTER_EPS = int(os.getenv("CLUSTER_EPS", "14"))
MIN_EDGE_LEN = float(os.getenv("MIN_EDGE_LEN", "0.20"))

TILE_W = int(os.getenv("TILE_W", "512"))

CONF_THRESH = float(os.getenv("CONF_THRESH", "0.50 "))
OVERLAP_THR = int(os.getenv("OVERLAP_THR", "60"))
NMS_IOU = float(os.getenv("NMS_IOU", "0.25"))

DW_CLASS_CONF = {
    "door": float(os.getenv("DOOR_CONF_THRESH", "0.40")),
    "window": float(os.getenv("WINDOW_CONF_THRESH", "0.30")),
}


OVERLAP = int(os.getenv("OVERLAP", "192"))
ALIGN_NEIGHBORHOOD_DEG = float(os.getenv("ALIGN_NEIGHBORHOOD_DEG", "10.0"))
ALIGN_VERTICAL_WEIGHT = float(os.getenv("ALIGN_VERTICAL_WEIGHT", "0.18"))
ALIGN_CENTER_PULL = float(os.getenv("ALIGN_CENTER_PULL", "0.08"))
OPENING_MIN_WALL_MARGIN = float(os.getenv("OPENING_MIN_WALL_MARGIN", "0.10"))

PANORAMA_SIZE = (1024, 512)

RF_CLASS_MAP_ID = {
    0: "window",
    1: "door",
    2: "door",
}
