"""Display live poses using a local camera and the Rockchip ONNX pose model."""
import argparse
from pathlib import Path
import time

import cv2
import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description='Display live human poses from a camera.')
parser.add_argument('--camera', type=int, default=0, help='Camera device index (default: 0)')
parser.add_argument('--backend', choices=('dshow', 'auto'), default='dshow', help='Windows camera backend')
parser.add_argument('--width', type=int, default=1280, help='Requested camera width')
parser.add_argument('--height', type=int, default=720, help='Requested camera height')
parser.add_argument('--person-confidence', type=float, default=0.5, help='Person detection threshold')
parser.add_argument('--keypoint-confidence', type=float, default=0.5, help='Keypoint drawing threshold')
parser.add_argument('--output', type=Path, default=ROOT / 'camera_pose.avi', help='Skeleton video output path (AVI; default: camera_pose.avi)')
parser.add_argument('--fps', type=float, default=15.0, help='Output video playback FPS (default: 15)')
parser.add_argument('--duration', type=float, default=None, help='Stop recording after this many seconds')
args = parser.parse_args()
if args.camera < 0 or args.width <= 0 or args.height <= 0:
    parser.error('Camera index must be nonnegative and dimensions must be positive')
if not 0 <= args.person_confidence <= 1 or not 0 <= args.keypoint_confidence <= 1:
    parser.error('Confidence thresholds must be between 0 and 1')
if not np.isfinite(args.fps) or args.fps <= 0:
    parser.error('Output FPS must be a positive finite number')
if args.duration is not None and (not np.isfinite(args.duration) or args.duration <= 0):
    parser.error('Duration must be a positive finite number')
if args.output.suffix.lower() != '.avi':
    parser.error('Output must use the .avi extension')
if not args.output.parent.is_dir():
    parser.error(f'Output directory does not exist: {args.output.parent}')
if args.output.exists():
    parser.error(f'Output already exists; choose another path: {args.output}')

# Reuse the matching Rockchip preprocessing and decoder without initializing RKNN.
source = (ROOT / 'official_yolov8_pose.py').read_text(encoding='utf-8')
source = source.replace('from rknn.api import RKNN', '')
pose = {'__name__': 'rockchip_pose'}
exec(compile(source, 'official_yolov8_pose.py', 'exec'), pose)
pose['objectThresh'] = args.person_confidence
session = ort.InferenceSession(str(ROOT / 'yolov8n-pose.onnx'), providers=['CPUExecutionProvider'])
backend = cv2.CAP_DSHOW if args.backend == 'dshow' else cv2.CAP_ANY
camera = cv2.VideoCapture(args.camera, backend)
if not camera.isOpened():
    camera.release()
    raise SystemExit(f'Cannot open camera {args.camera}. Check permissions or try another index/backend.')
camera.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
print(f'Camera {args.camera} opened; requested {args.width}x{args.height}. Press Q or Esc to quit.', flush=True)
writer = None
frames_written = 0
recording_started = None
try:
    while True:
        ok, frame = camera.read()
        if not ok or frame is None:
            print('Camera frame unavailable; stopping.', flush=True)
            break
        start = time.perf_counter()
        resized, scale, dx, dy = pose['letterbox_resize'](frame, (640, 640), 56)
        data = np.ascontiguousarray(resized[:, :, ::-1].transpose(2, 0, 1)[None], dtype=np.float32) / 255.0
        outputs = session.run(None, {session.get_inputs()[0].name: data})
        boxes = []
        for output in outputs[:3]:
            height, width = output.shape[2:]
            boxes.extend(pose['process'](output.reshape(1, 65, -1), outputs[3], {80: 0, 40: 6400, 20: 8000}[height], width, height, 640 // height))
        boxes = pose['NMS'](boxes)
        for box in boxes:
            coords = [int((box.xmin - dx) / scale), int((box.ymin - dy) / scale), int((box.xmax - dx) / scale), int((box.ymax - dy) / scale)]
            points = box.keypoint.reshape(-1, 3).copy()
            points[:, 0] = (points[:, 0] - dx) / scale
            points[:, 1] = (points[:, 1] - dy) / scale
            cv2.rectangle(frame, tuple(coords[:2]), tuple(coords[2:]), (0, 255, 0), 2)
            cv2.putText(frame, f'person {box.score:.2f}', (coords[0], max(20, coords[1] - 5)), cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 255, 0), 2)
            for i, (x, y, confidence) in enumerate(points):
                if confidence >= args.keypoint_confidence:
                    cv2.circle(frame, (int(x), int(y)), 4, [int(c) for c in pose['kpt_color'][i]], -1, cv2.LINE_AA)
            for i, (a, b) in enumerate(pose['skeleton']):
                if points[a - 1, 2] >= args.keypoint_confidence and points[b - 1, 2] >= args.keypoint_confidence:
                    cv2.line(frame, tuple(points[a - 1, :2].astype(int)), tuple(points[b - 1, :2].astype(int)), [int(c) for c in pose['limb_color'][i]], 2, cv2.LINE_AA)
        elapsed_ms = (time.perf_counter() - start) * 1000
        cv2.putText(frame, f'People: {len(boxes)}  Processing: {elapsed_ms:.0f} ms', (10, 25), cv2.FONT_HERSHEY_SIMPLEX, .65, (0, 255, 0), 2)
        if writer is None:
            height, width = frame.shape[:2]
            writer = cv2.VideoWriter(str(args.output.resolve()), cv2.VideoWriter_fourcc(*'MJPG'), args.fps, (width, height))
            if not writer.isOpened():
                raise RuntimeError(f'Cannot create video: {args.output}. Try an ASCII-only path.')
            recording_started = time.monotonic()
            print(f'Recording {width}x{height} at playback {args.fps:g} FPS to {args.output.resolve()}', flush=True)
        writer.write(frame)
        frames_written += 1
        cv2.imshow('YOLOv8 Pose - press Q to quit', frame)
        key = cv2.waitKey(1) & 0xff
        if (args.duration is not None and time.monotonic() - recording_started >= args.duration) or key in (ord('q'), ord('Q'), 27) or cv2.getWindowProperty('YOLOv8 Pose - press Q to quit', cv2.WND_PROP_VISIBLE) < 1:
            break
finally:
    camera.release()
    if writer is not None:
        writer.release()
        duration = time.monotonic() - recording_started
        print(f'Saved {frames_written} frames over {duration:.1f} seconds to {args.output.resolve()}', flush=True)
        print('Video uses a fixed playback FPS; elapsed time may differ from recording time.', flush=True)
    cv2.destroyAllWindows()
