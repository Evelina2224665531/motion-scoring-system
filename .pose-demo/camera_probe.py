"""Probe camera indices, reading one frame and releasing each device."""
import cv2

for name, backend in [('DirectShow', cv2.CAP_DSHOW), ('Media Foundation', cv2.CAP_MSMF)]:
    for index in range(5):
        camera = cv2.VideoCapture(index, backend)
        opened = camera.isOpened()
        ok, frame = camera.read() if opened else (False, None)
        print(f'{name} index={index}: opened={opened}, frame={ok}, shape={None if frame is None else frame.shape}', flush=True)
        camera.release()
