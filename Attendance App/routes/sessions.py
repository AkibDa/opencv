from flask import Blueprint, request, jsonify
from db import mongo
from bson import ObjectId
from datetime import datetime

sessions_bp = Blueprint("sessions", __name__)

@sessions_bp.post("/")
def create_session():
    data = request.json
    try:
        starts_at = datetime.fromisoformat(data["starts_at"])
        ends_at = datetime.fromisoformat(data["ends_at"])
    except Exception:
        return jsonify({"ok": False, "msg": "Invalid date format"}), 400

    if starts_at >= ends_at:
        return jsonify({"ok": False, "msg": "Start time must be before end time"}), 400

    session = {
        "class_id": ObjectId(data["class_id"]),
        "starts_at": starts_at,
        "ends_at": ends_at,
        "room": data.get("room", ""),
        "created_at": datetime.utcnow()
    }

    result = mongo.db.sessions.insert_one(session)
    return jsonify({"ok": True, "msg": "Session created", "session_id": str(result.inserted_id)})
