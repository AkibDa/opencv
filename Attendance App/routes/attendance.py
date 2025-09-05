from flask import Blueprint, request, jsonify
from db import mongo
from bson import ObjectId
from ai.recognizer import Recognizer
import cv2, base64, numpy as np
from datetime import datetime

attendance_bp = Blueprint("attendance", __name__)
rec = Recognizer()

@attendance_bp.post("/recognize")
def attendance_recognize():
    data = request.json
    session_id = ObjectId(data["session_id"])
    img_b64 = data["image"]
    threshold = data.get("threshold", 0.42)

    img = cv2.imdecode(np.frombuffer(base64.b64decode(img_b64), np.uint8), cv2.IMREAD_COLOR)
    dets = rec.detect_embed(img)
    if not dets:
        return jsonify({"ok": False, "msg": "No faces detected"}), 400

    # Prepare gallery from students
    gallery = []
    students = mongo.db.students.find({})
    for s in students:
        emb = s.get("face_embedding")
        if emb and len(emb) > 0:
            gallery.append((s["_id"], np.array(emb)))

    results = []
    for bbox, emb, _ in dets:
        emb = emb.astype(float)
        best_uid, best_sim = rec.match(emb, gallery, threshold=threshold)

        if best_uid:
            mongo.db.attendance.update_one(
                {"session_id": session_id, "student_id": best_uid},
                {"$set": {
                    "status": "present",
                    "confidence": float(best_sim),
                    "method": "face",
                    "marked_at": datetime.utcnow()
                }},
                upsert=True
            )

        results.append({
            "bbox": [int(x) for x in bbox],
            "user_id": str(best_uid) if best_uid else None,
            "score": best_sim
        })

    return jsonify({"ok": True, "results": results})
