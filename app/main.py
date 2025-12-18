from fastapi.responses import HTMLResponse
from fastapi import FastAPI, UploadFile, File
import shutil, os
from app.video_processor import analyze_video

app = FastAPI(title="Poultry Video Analytics")

@app.get("/")
def root():
    return {"message": "Go to /ui for the interface"}

@app.get("/ui", response_class=HTMLResponse)
def ui():
    with open("app/ui.html", "r", encoding="utf-8") as f:
        return f.read()

@app.get("/health")
def health():
    return {"status": "OK"}

@app.post("/analyze_video")
async def analyze(
    video: UploadFile = File(...),
    fps_sample: int = 5,
    conf_thresh: float = 0.4,
    iou_thresh: float = 0.5
):
    os.makedirs("inputs", exist_ok=True)
    video_path = f"inputs/{video.filename}"

    with open(video_path, "wb") as f:
        shutil.copyfileobj(video.file, f)

    return analyze_video(video_path, fps_sample, conf_thresh, iou_thresh)
