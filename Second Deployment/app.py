# app.py
import os
import uuid
from flask import Flask, request, jsonify
import torch

from model import load_model
from utils import video_to_tensor_3d

# ---------- Config ----------
UPLOAD_DIR = "uploads"
ALLOWED_EXT = ".mp4"

os.makedirs(UPLOAD_DIR, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = load_model("../Resources/cnn3d_best1.pth", device=device)

CLASS_NAMES = {0: "non_lifter", 1: "lifter"}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024

def is_allowed(filename: str) -> bool:
    ext = os.path.splitext(filename)[1].lower()
    return ext == ALLOWED_EXT

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "device": device})

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file part. Use multipart/form-data with 'file'."}), 400

    f = request.files["file"]
    if f.filename == "":
        return jsonify({"error": "Empty filename"}), 400
    if not is_allowed(f.filename):
        return jsonify({"error": f"Unsupported extension. Allowed: {sorted(ALLOWED_EXT)}"}), 400

    uid = uuid.uuid4().hex
    ext = os.path.splitext(f.filename)[1].lower()
    save_path = os.path.join(UPLOAD_DIR, f"{uid}{ext}")
    f.save(save_path)

    try:
        with torch.no_grad():
            clip = video_to_tensor_3d(save_path, device=device)  # (1, C, T, H, W)
            logits = model(clip)
            probs = torch.softmax(logits, dim=1).cpu().numpy().tolist()[0]
            pred_idx = int(torch.argmax(logits, dim=1).item())
            pred_label = CLASS_NAMES.get(pred_idx, str(pred_idx))

        resp = {
            "prediction": pred_label,
            "class_index": pred_idx,
            "probabilities": {
                "non_lifter": float(probs[0]),
                "lifter": float(probs[1])
            }
        }
        return jsonify(resp), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        try:
            os.remove(save_path)
        except Exception:
            pass

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True, threaded=True)