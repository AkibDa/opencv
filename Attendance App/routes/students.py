from flask import Blueprint, request, jsonify
from db import mongo
from datetime import datetime
from bson import ObjectId

students_bp = Blueprint("students", __name__)

@students_bp.post("/")
def add_student():
    data = request.json
    student = {
        "name": data["name"],
        "roll_no": data["roll_no"],
        "email": data["email"],
        "photo_url": data.get("photo_url", ""),
        "face_embedding": data.get("face_embedding", []),
        "created_at": datetime.utcnow()
    }
    result = mongo.db.students.insert_one(student)
    return jsonify({"ok": True, "student_id": str(result.inserted_id)})
