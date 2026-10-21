# collect_negatives.py
from pathlib import Path
import time
import cv2
import numpy as np
from utils import get_face_embedding

# === Paths ===
BASE = Path(__file__).resolve().parents[1] / "data"
BASE.mkdir(parents=True, exist_ok=True)
NEG_PATH = BASE / "negatives_A.npy"

# === Camera ===
cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)

if not cap.isOpened():
    raise RuntimeError("Cannot open camera")

if NEG_PATH.exists():
    all_embs = list(np.load(NEG_PATH))
    print(f"[INFO] Existing neg {len(all_embs)}")
else:
    all_embs = []


while True:
    ok, frame = cap.read()
    if not ok:
        print("[WARN] No frame ")
        break

    emb = get_face_embedding(frame)
    msg = f"total neg: {len(all_embs)}"
    color = (0, 0, 255) if emb is not None else (0, 255, 255)

    cv2.putText(frame, msg, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    cv2.putText(frame, "SPACE=save  ESC=quit", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    cv2.imshow("collect_negatives", frame)

    k = cv2.waitKey(1) & 0xFF
    if k == 27:   # ESC
        break
    elif k == 32: # SPACE
        if emb is not None:
            all_embs.append(emb.astype(np.float32))
            print(f"[+] Save #{len(all_embs)}")
            time.sleep(0.15)
        else:
            print("[-] Nothing.")

cap.release()
cv2.destroyAllWindows()

if all_embs:
    np.save(NEG_PATH, np.stack(all_embs))
    print(f"[OK] Savesd {len(all_embs)} embeddings in {NEG_PATH}")
else:
    print("Nothing new")
