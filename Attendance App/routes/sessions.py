from flask import Blueprint, request, jsonify
from db import mongo
from bson import ObjectId
from datetime import datetime

sessions_bp = Blueprint("sessions", __name__)

@sessions_bp.post("/sessions")
def create_session():
  data = request.json
  session = {
    "class_id": ObjectId(data["class_id"]),
    "starts_at": datetime.fromisoformat(data["starts_at"]),
    "ends_at": datetime.fromisoformat(data["ends_at"]),
    "room": data.get("room", ""),
    "created_at": datetime.utcnow()
  }
  mongo.db.sessions.insert_one(session)
  return jsonify({"ok": True, "msg": "Session created"})
