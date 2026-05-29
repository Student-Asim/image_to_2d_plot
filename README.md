# MagicPlan 2D Floor Plan API

A FastAPI-based project that converts panorama or room images into a structured 2D floor plan with detected walls, doors, windows, and furniture. The pipeline combines HorizonNet-style room layout estimation, Roboflow-based door/window detection, and Ultralytics YOLO furniture detection, then exports both PNG and DXF outputs. 

## Features

- Generates a simplified 2D room/floor layout from panoramic room imagery.
- Detects doors and windows and maps them onto wall segments.
- Detects furniture objects such as beds, sofas, chairs, and tables.
- Exports results as rendered floor plan images and DXF files.
- Provides a FastAPI endpoint for uploading an image and receiving generated outputs. [web:164][web:186]

## Project structure

```text
MagicPlan_2D_floor_plan/
├── api_endpoint/
│   ├── config.py
│   ├── main.py
│   ├── pipeline.py
│   ├── model.pth
│   └── outputs/
├── HorizonNet/
│   ├── model.py
│   └── ...
├── requirements.txt
└── README.md
```

### Main components

- `main.py` — FastAPI application and API routes.
- `config.py` — central configuration using environment variables and defaults.
- `pipeline.py` — end-to-end inference pipeline for layout, openings, and furniture.
- `model.pth` — HorizonNet weights file.
- `HorizonNet/model.py` — Python source that defines the `HorizonNet` architecture.
- `outputs/` — generated PNG, DXF, and debug files. [web:184][web:190]

## How it works

### 1. Layout estimation

The project loads a HorizonNet model, recreates the architecture in Python, and then loads the saved weights from a `.pth` file. This is the standard PyTorch pattern: create the model class first, then load the `state_dict` using `load_state_dict()`. [web:184][web:190]

### 2. Door and window detection

Door and window candidates are detected through the Roboflow API. The detected boxes are normalized, filtered, and aligned to the estimated room walls before being drawn into the final 2D plan.

### 3. Furniture detection

Furniture is detected using Ultralytics YOLO in Python with a model loaded through `YOLO(...)`. The detections are filtered by class and confidence, then mapped into the room polygon as simplified 2D furniture symbols. [web:186][web:189]

### 4. Rendering and export

The final plan is rendered as a floor-plan style PNG and exported as DXF for CAD workflows. Debug overlays can also be generated for layout, openings, and furniture placement.

## Requirements

This project uses Python 3.11 and the following main dependencies:

```txt
fastapi
uvicorn
python-multipart
pillow
opencv-python
ezdxf
matplotlib
numpy
requests
torch
torchvision
scipy
shapely
scikit-learn
ultralytics
```

Ultralytics supports Python integration through the `ultralytics` package, while FastAPI apps are typically served with an ASGI server such as Uvicorn. [web:186][web:164]

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/MagicPlan_2D_floor_plan.git
cd MagicPlan_2D_floor_plan
```

### 2. Create and activate a virtual environment

#### Windows

```bash
python -m venv venv
venv\Scripts\activate
```


### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Verify required files

Make sure these files exist before starting the app:

- `api_endpoint/model.pth` or the configured HorizonNet weights path
- `HorizonNet/model.py`
- FastAPI source files inside `api_endpoint/`

## Configuration

Configuration is handled in `api_endpoint/config.py` with environment variable overrides.

### Important configuration values

| Variable | Purpose |
|---|---|
| `YOLO_MODEL` | Ultralytics model name or path, e.g. `yolov8x.pt` |
| `HORIZONNET_WEIGHTS` | Path to the `.pth` weights file (`api_endpoint/model.pth` by default) |
| `HORIZONNET_API_KEY` | Currently loaded from env but not used by the current pipeline |
| `ROBOFLOW_API_KEY` | Roboflow API key required for door/window detection |
| `ROBOFLOW_URL` | Roboflow inference endpoint |
| `OUTPUT_DIR` | Output folder for generated files |
| `ROOM_SCALE` | Scale applied to normalized room geometry |
| `WALL_THICKNESS` | Floor plan wall thickness |
| `DOOR_WIDTH_M` | Default door width in meters |
| `WINDOW_WIDTH_M` | Default window width in meters |

### Recommended local config

By default, `api_endpoint/config.py` uses the local weights file at `api_endpoint/model.pth`:

```python
ROOT_DIR = Path(__file__).resolve().parent
HORIZONNET_WEIGHTS = Path(os.getenv("HORIZONNET_WEIGHTS", str(ROOT_DIR / "model.pth"))).resolve()
```

If you want to override the default, set `HORIZONNET_WEIGHTS` to the desired `.pth` path.

### Security note

Do not hardcode API keys in source code for production or public repositories. Store them in environment variables instead. FastAPI commonly uses environment-based settings for deployment configuration. [web:181]

## Correct imports

A common source of errors in this project is mixing up the `.pth` weights file with the Python model definition.

### Correct idea

- `model.pth` = trained weights
- `HorizonNet/model.py` = Python code defining `HorizonNet`

### Correct import in `pipeline.py`

```python
from HorizonNet.model import HorizonNet
```

### Correct model loading pattern

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = HorizonNet(backbone="resnet50", use_rnn=True).to(device)
ckpt = torch.load(C.HORIZONNET_WEIGHTS, map_location=device)
model.load_state_dict(ckpt.get("state_dict", ckpt), strict=True)
model.eval()
```

PyTorch recommends recreating the model instance and then loading weights through `load_state_dict()`. [web:184][web:190]

## Running the API

Start the server from the project root so Python can resolve both `api_endpoint` and `HorizonNet` properly.

```bash
python -m uvicorn api_endpoint.main:app --reload --host 0.0.0.0 --port 8000
```

FastAPI applications are typically served by an ASGI server such as Uvicorn, and development mode often uses `--reload`. For production, FastAPI recommends running without `--reload`. [web:164][web:180]

## API usage

### Swagger UI

After the server starts, open:

```text
http://127.0.0.1:8000/docs
```

FastAPI automatically provides interactive API documentation through Swagger UI. [web:191]

### Example request with cURL

```bash
curl -X POST "http://127.0.0.1:8000/your-endpoint" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@your_image.jpg"
```

Replace `/your-endpoint` with the actual upload route defined in `main.py`.

## Outputs

The pipeline can generate several output files in `outputs/`:

- `*_floorplan.png` — final rendered floor plan
- `*_floorplan.dxf` — CAD-friendly DXF export
- `*_layout_debug.png` — layout debug visualization
- `*_openings_debug.png` — door/window overlay
- `*_furniture_debug.png` — furniture detection overlay

These files help debug each pipeline stage independently.

## Common errors

### 1. `ModuleNotFoundError: No module named 'model'`

Cause:
`from model import HorizonNet` was used, but there is no `model.py` beside `pipeline.py`.

Fix:
Use the actual source package path:

```python
from HorizonNet.model import HorizonNet
```

The `.pth` file is not importable Python code. It is only a weights/checkpoint file. [web:165][web:170]

### 2. `ImportError: attempted relative import with no known parent package`

Cause:
The app was started from the wrong folder, or package-style imports were mixed with script-style execution.

Fix:
Run the app from the project root:

```bash
python -m uvicorn api_endpoint.main:app --reload
```

### 3. Weights file loads but model import fails

Cause:
The project has the `.pth` file but not the Python source defining the architecture.

Fix:
Ensure the HorizonNet source code exists and is importable, for example:

```text
HorizonNet/model.py
```

PyTorch weight loading requires the architecture class to exist in Python before the weights can be applied. [web:184][web:190]

### 4. Roboflow request failures

Cause:
Missing API key, invalid endpoint, timeout, or connectivity issue.

Fix:
Check:

- `ROBOFLOW_API_KEY`
- `ROBOFLOW_URL`
- timeout and retry settings in `config.py`

## Example `requirements.txt`

```txt
fastapi==0.115.0
uvicorn[standard]==0.30.6
ultralytics==8.3.6
opencv-python==4.10.0.84
ezdxf==1.3.5
pillow==10.4.0
numpy==1.26.4
python-multipart==0.0.9
python-dotenv>=0.21.0
torch>=2.0
torchvision>=0.15
matplotlib>=3.8
requests>=2.31
scipy>=1.11
shapely>=2.0
scikit-learn>=1.4
```

## Development notes

- Keep imports consistent with the real folder structure.
- Do not confuse `model.py` with `model.pth`.
- Run the app from the repository root.
- Keep secrets in environment variables instead of source files.
- Use debug output images to verify each stage of the pipeline.

## Future improvements

- Add Docker support for easier deployment.
- Add unit tests for geometry and mapping utilities.
- Add request/response schema documentation.
- Add batch processing for multiple panorama images.
- Add persistent storage for processed jobs and artifacts.

## License

