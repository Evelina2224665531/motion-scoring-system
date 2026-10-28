"""Run Rockchip's example model with ONNX Runtime and its pose postprocessing."""
from pathlib import Path
import argparse
import time
import cv2
import numpy as np
import onnxruntime as ort

root = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description='Detect human poses in an image with the Rockchip ONNX model.')
parser.add_argument('--image', type=Path, default=root / 'bus.jpg', help='Input image path (default: bundled bus.jpg)')
parser.add_argument('--output', type=Path, default=root / 'result.jpg', help='Output image path (default: result.jpg)')
args = parser.parse_args()
if not args.image.is_file():
    parser.error(f'Input image does not exist: {args.image}')
if args.image.resolve() == args.output.resolve():
    parser.error('Output image must be different from the input image')
source = (root / 'official_yolov8_pose.py').read_text(encoding='utf-8')
source = source.replace('from rknn.api import RKNN', '')
namespace = {'__name__': 'rockchip_pose'}
exec(compile(source, 'official_yolov8_pose.py', 'exec'), namespace)

image = cv2.imdecode(np.fromfile(args.image, dtype=np.uint8), cv2.IMREAD_COLOR)
if image is None:
    raise RuntimeError(f'Cannot decode input image: {args.image}')
letterbox, scale, offset_x, offset_y = namespace['letterbox_resize'](image, (640, 640), 56)
tensor = np.ascontiguousarray(letterbox[:, :, ::-1].transpose(2, 0, 1)[None], dtype=np.float32) / 255.0
session = ort.InferenceSession(str(root / 'yolov8n-pose.onnx'), providers=['CPUExecutionProvider'])
start = time.perf_counter()
results = session.run(None, {session.get_inputs()[0].name: tensor})
elapsed = (time.perf_counter() - start) * 1000
boxes = []
for output in results[:3]:
    height, width = output.shape[2:]
    stride = 640 // height
    index = {80: 0, 40: 6400, 20: 8000}[height]
    boxes.extend(namespace['process'](output.reshape(1, 65, -1), results[3], index, width, height, stride))
boxes = namespace['NMS'](boxes)
print(f'Image: {image.shape[1]} x {image.shape[0]}')
print(f'Provider: {session.get_providers()}')
print(f'ONNX inference: {elapsed:.1f} ms (single run, no warm-up)')
print(f'People detected: {len(boxes)}')
for i, box in enumerate(boxes, 1):
    coords = [int((box.xmin-offset_x)/scale), int((box.ymin-offset_y)/scale), int((box.xmax-offset_x)/scale), int((box.ymax-offset_y)/scale)]
    points = box.keypoint.reshape(-1, 3).copy()
    points[:, 0] = (points[:, 0] - offset_x) / scale
    points[:, 1] = (points[:, 1] - offset_y) / scale
    print(f'person {i}: confidence={box.score:.3f}, box={coords}, keypoints above 0.5={int((points[:, 2] >= 0.5).sum())}/17')
    cv2.rectangle(image, tuple(coords[:2]), tuple(coords[2:]), (0, 255, 0), 2)
    cv2.putText(image, f'person {box.score:.2f}', tuple(coords[:2]), cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 0, 255), 2)
    for j, (x, y, confidence) in enumerate(points):
        if confidence >= 0.5:
            cv2.circle(image, (int(x), int(y)), 4, [int(c) for c in namespace['kpt_color'][j]], -1, cv2.LINE_AA)
    for j, (a, b) in enumerate(namespace['skeleton']):
        if points[a-1, 2] >= 0.5 and points[b-1, 2] >= 0.5:
            cv2.line(image, tuple(points[a-1, :2].astype(int)), tuple(points[b-1, :2].astype(int)), [int(c) for c in namespace['limb_color'][j]], 2, cv2.LINE_AA)
output_path = args.output.resolve()
if not output_path.parent.is_dir():
    parser.error(f'Output directory does not exist: {output_path.parent}')
success, encoded = cv2.imencode(output_path.suffix, image)
if not success:
    raise RuntimeError(f'Cannot encode output image: {output_path}')
encoded.tofile(output_path)
print(f'Output: {output_path}')
