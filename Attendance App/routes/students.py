from flask import Blueprint, request, jsonify
from db import mongo
from datetime import datetime

students_bp = Blueprint("students", __name__)

@students_bp.post("/students")
def add_student():
  data = request.json
  student = {
    "name": data["name"],
    "roll_no": data["roll_no"],
    "email": data["email"],
    "photo_url": data.get("photo_url", ""),
    "created_at": datetime.utcnow()
  }
  mongo.db.students.insert_one(student)
  return jsonify({"ok": True, "student": str(student)})
