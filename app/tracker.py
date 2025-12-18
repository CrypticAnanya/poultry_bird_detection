import itertools

class SimpleTracker:
    def __init__(self, iou_thresh=0.5):
        self.tracks = {}
        self.next_id = itertools.count(1)
        self.iou_thresh = iou_thresh

    def update(self, detections):
        updated_tracks = []

        for det in detections:
            tid = next(self.next_id)
            bbox = det[:4]
            self.tracks[tid] = bbox
            updated_tracks.append({
                "id": tid,
                "bbox": bbox
            })

        return updated_tracks
