# FaceID-Desk

Academic  project (2025) exploring 1:1 face verification with blink-based liveness checks. The prototype uses Python, OpenCV, InsightFace/ArcFace, and MediaPipe.

## What it does

- Captures enrollment samples from a webcam and creates a local face embedding template.
- Collects negative samples and calibrates a similarity threshold.
- Verifies a live face, with an optional blink check as a basic liveness signal.

This is an educational prototype, not a secure authentication system. Blink checks can be bypassed, and accuracy depends on lighting, camera quality, and the chosen threshold.

## Setup

Use Python 3.10 or 3.11 in a virtual environment, then install the packages listed in `requirements.txt`. InsightFace may download its model files on first run.

Run the scripts from this directory in this order:

1. `python jtaboad/enroll.py`
2. `python jtaboad/collect_negatives.py`
3. `python jtaboad/calibrate.py`
4. `python jtaboad/verify.py` or `python jtaboad/verify_blink.py`

The scripts expect a `data/` folder in this repository. They create it when collecting enrollment or negative samples.

## Biometric data

Enrollment templates and face embeddings are sensitive biometric data. They are generated locally and must stay out of Git. `.gitignore` excludes the generated `data/` files. Do not commit face images, embeddings, templates, or recordings.

## License and models

No license has been selected for this repository. InsightFace and its pretrained models have their own terms; review those terms before redistribution or commercial use.
