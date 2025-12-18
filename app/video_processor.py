import cv2
import os
from ultralytics import YOLO
from app.tracker import SimpleTracker
from app.weight import estimate_weight_index

def analyze_video(video_path, fps_sample, conf_thresh, iou_thresh):
    model = YOLO("yolov8n.pt")
    tracker = SimpleTracker(iou_thresh=iou_thresh)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    os.makedirs("outputs", exist_ok=True)
    output_path = "outputs/annotated_output.mp4"

    writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height)
    )

    frame_id = 0
    counts = []
    weight_logs = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_id % fps_sample != 0:
            frame_id += 1
            continue

        results = model(frame, conf=conf_thresh)[0]
        detections = []

        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            detections.append([x1, y1, x2, y2, conf])

        tracks = tracker.update(detections)

        timestamp = round(frame_id / fps, 2)
        counts.append({"time": timestamp, "count": len(tracks)})

        for track in tracks:
            x1, y1, x2, y2 = track["bbox"]
            tid = track["id"]

            weight_idx = estimate_weight_index(track["bbox"])
            weight_logs.append({
                "track_id": tid,
                "time": timestamp,
                "weight_index": weight_idx
            })

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(frame, f"ID:{tid}", (x1, y1-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

        cv2.putText(frame, f"Count: {len(tracks)}", (20,40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

        writer.write(frame)
        frame_id += 1

    cap.release()
    writer.release()

    return {
        "counts": counts,
        "tracks_sample": tracks[:5] if tracks else [],
        "weight_estimates": {
            "unit": "relative_weight_index",
            "values": weight_logs,
            "note": "Calibration required to convert to grams"
        },
        "artifacts": {
            "annotated_video": output_path
        }
    }
