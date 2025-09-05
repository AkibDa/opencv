from flask import Blueprint, request, jsonify
from db import mongo
from datetime import datetime

teachers_bp = Blueprint("teachers", __name__)

@teachers_bp.post("/teachers")
def add_teacher():
  data = request.json
  teacher = {
    "name": data["name"],
    "emp_id": data["emp_id"],
    "email": data["email"],
    "department": data.get("department", ""),
    "created_at": datetime.utcnow()
  }
  mongo.db.teachers.insert_one(teacher)
  return jsonify({"ok": True, "teacher": str(teacher)})
