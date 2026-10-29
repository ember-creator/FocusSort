# Third-party notices

FocusSort bundles or uses the following open-source components and models.

## OpenCV

- Project: https://opencv.org/
- License: Apache License 2.0
- Used for image metrics, face detection post-processing, Haar cascades, and the TFLite importer.
- Bundled under `vendor/cv2/`.

## BlazeFace face detection model

- Model: `face_detection_short_range.tflite`
- Source: Google MediaPipe / OpenCV Zoo
  - https://github.com/google-ai-edge/mediapipe
  - https://github.com/opencv/opencv_zoo
- License: Apache License 2.0
- The model is used only to locate faces. No face recognition embeddings or identity templates are produced or stored.

## TkinterDnD2 / tkdnd

- Project: https://github.com/pmgagne/tkinterdnd2
- License: MIT (TkinterDnD2); tkdnd is distributed under its own permissive license.
- Used for drag-and-drop file import.

## Pillow

- Project: https://python-pillow.org/
- License: HPND / PIL license.
- Used for image decoding, EXIF orientation, thumbnails, and the user interface.

## NumPy

- Project: https://numpy.org/
- License: BSD-3-Clause.
- Used for numerical operations.

## pillow-heif (optional)

- Project: https://github.com/bigcat88/pillow_heif
- License: BSD-3-Clause (with bundled libheif under LGPL-3.0).
- Not bundled in the clean ZIP. Install it to decode HEIC/HEIF:
  `python -m pip install pillow-heif`.
- Used only for local HEIC/HEIF decoding. Without it, the UI shows a clear
  decoder-missing message and suggests conversion to JPG.

## Fonts

The application uses the system UI font (Microsoft YaHei UI / Segoe UI on Windows). No font files are redistributed.

## Sample images

`docs/screenshot.png` contains a screenshot of the application using public-domain NASA astronaut imagery and a U.S. Navy portrait (Grace Hopper) for demonstration. These images are not part of the analysis pipeline and are not required at runtime.
