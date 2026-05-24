import io
import logging
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from PIL import Image, UnidentifiedImageError

import config as C
import pipeline as P

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s %(message)s",
)
log = logging.getLogger("main")

_jobs: dict[str, dict] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Loading models...")
    P.load_models()
    log.info("Models ready.")
    yield
    log.info("Shutdown complete.")


app = FastAPI(
    title="2D CAD Floor Plan API",
    description=(
        "Upload a 360 panorama and get a clean 2D floor plan with wall polygon, "
        "plus door/window alignment from panorama detections."
    ),
    version="2.0.0",
    lifespan=lifespan,
)

ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/jpg"}


@app.get("/health", tags=["System"])
def health():
    return {"status": "ok"}


@app.post("/generate", tags=["Floor Plan"])
async def generate_floorplan(file: UploadFile = File(...)):
    image_bytes = await _read_and_validate_image(file)
    job_id = uuid.uuid4().hex

    try:
        result = P.run_pipeline(image_bytes, job_id)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        log.exception("Job %s failed", job_id)
        raise HTTPException(status_code=500, detail=f"Pipeline error: {e}")

    _jobs[job_id] = {"status": "done", **result}
    return {
        **result,
        "download_png": f"/download/{job_id}/png",
        "download_dxf": f"/download/{job_id}/dxf",
        "debug_layout_png": f"/download/{job_id}/layout-debug",
        "debug_openings_png": f"/download/{job_id}/openings-debug",
    }


@app.post("/generate/async", tags=["Floor Plan"], status_code=202)
async def generate_floorplan_async(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
):
    image_bytes = await _read_and_validate_image(file)
    job_id = uuid.uuid4().hex
    _jobs[job_id] = {"status": "processing"}
    background_tasks.add_task(_run_background, image_bytes, job_id)
    return {
        "job_id": job_id,
        "status": "processing",
        "poll_url": f"/jobs/{job_id}",
    }


@app.get("/jobs/{job_id}", tags=["Floor Plan"])
def job_status(job_id: str):
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    response = {**job}
    if job["status"] == "done":
        response["download_png"] = f"/download/{job_id}/png"
        response["download_dxf"] = f"/download/{job_id}/dxf"
        response["debug_layout_png"] = f"/download/{job_id}/layout-debug"
        response["debug_openings_png"] = f"/download/{job_id}/openings-debug"
    return response


@app.get("/download/{job_id}/png", tags=["Download"])
def download_png(job_id: str):
    path = _get_output_path(job_id, "png")
    return FileResponse(
        str(path),
        media_type="image/png",
        filename=f"floorplan_{job_id}.png",
    )


@app.get("/download/{job_id}/dxf", tags=["Download"])
def download_dxf(job_id: str):
    path = _get_output_path(job_id, "dxf")
    return FileResponse(
        str(path),
        media_type="application/octet-stream",
        filename=f"floorplan_{job_id}.dxf",
    )


@app.get("/download/{job_id}/layout-debug", tags=["Download"])
def download_layout_debug(job_id: str):
    path = _get_named_output(job_id, "layout_debug_png_path")
    return FileResponse(
        str(path),
        media_type="image/png",
        filename=f"layout_debug_{job_id}.png",
    )


@app.get("/download/{job_id}/openings-debug", tags=["Download"])
def download_openings_debug(job_id: str):
    path = _get_named_output(job_id, "openings_debug_png_path")
    return FileResponse(
        str(path),
        media_type="image/png",
        filename=f"openings_debug_{job_id}.png",
    )


@app.get("/jobs", tags=["System"])
def list_jobs():
    return [{"job_id": jid, "status": info.get("status")} for jid, info in _jobs.items()]


async def _read_and_validate_image(file: UploadFile) -> bytes:
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(status_code=415, detail="Upload a JPEG or PNG panorama.")

    image_bytes = await file.read()

    try:
        with Image.open(io.BytesIO(image_bytes)) as im:
            im.verify()
    except (UnidentifiedImageError, OSError, SyntaxError):
        raise HTTPException(status_code=422, detail="Invalid image file.")

    return image_bytes


def _run_background(image_bytes: bytes, job_id: str):
    try:
        result = P.run_pipeline(image_bytes, job_id)
        _jobs[job_id] = {"status": "done", **result}
    except Exception as e:
        log.exception("Background job %s failed", job_id)
        _jobs[job_id] = {"status": "error", "detail": str(e)}


def _get_job(job_id: str) -> dict:
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job["status"] == "processing":
        raise HTTPException(status_code=202, detail="Job still processing")
    if job["status"] == "error":
        raise HTTPException(status_code=500, detail=job.get("detail", "Pipeline error"))
    return job


def _get_output_path(job_id: str, ext: str) -> Path:
    _get_job(job_id)
    path = C.OUTPUT_DIR / f"{job_id}_floorplan.{ext}"
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"{ext.upper()} file not found")
    return path


def _get_named_output(job_id: str, key: str) -> Path:
    job = _get_job(job_id)
    raw = job.get(key)
    if not raw:
        raise HTTPException(status_code=404, detail="Debug file not found")

    path = Path(raw)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Debug file not found")
    return path