from flask import Blueprint, request, jsonify
from db import mongo
from bson import ObjectId
from ai.recognizer import Recognizer
import cv2, base64, numpy as np
from datetime import datetime

enroll_bp = Blueprint("enroll", __name__)
rec = Recognizer()

@enroll_bp.post("/face")
def enroll_face():
    data = request.json
    user_id = data["user_id"]
    role = data["role"]  # "student" or "teacher"
    img_b64 = data["image"]

    img = cv2.imdecode(np.frombuffer(base64.b64decode(img_b64), np.uint8), cv2.IMREAD_COLOR)
    dets = rec.detect_embed(img)
    if not dets:
        return jsonify({"ok": False, "msg": "No face detected"}), 400

    emb = dets[0][1].astype(float).tolist()
    collection = mongo.db.students if role == "student" else mongo.db.teachers

    result = collection.update_one(
        {"_id": ObjectId(user_id)},
        {"$set": {"face_embedding": emb, "updated_at": datetime.utcnow()}}
    )

    if result.matched_count == 0:
        return jsonify({"ok": False, "msg": "User not found"}), 404

    return jsonify({"ok": True, "msg": "Face enrolled"})
