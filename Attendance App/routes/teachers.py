from flask import Blueprint, request, jsonify
from db import mongo
from datetime import datetime

teachers_bp = Blueprint("teachers", __name__)

@teachers_bp.get("/")
def list_teachers():
    teachers = list(mongo.db.teachers.find({}, {"face_embedding": 0}))
    for t in teachers:
        t["_id"] = str(t["_id"])
    return jsonify({"ok": True, "teachers": teachers})


@teachers_bp.post("/")
def add_teacher():
    data = request.json
    teacher = {
        "name": data["name"],
        "emp_id": data["emp_id"],
        "email": data["email"],
        "department": data.get("department", ""),
        "face_embedding": data.get("face_embedding", []),
        "created_at": datetime.utcnow()
    }
    mongo.db.teachers.insert_one(teacher)
    return jsonify({"ok": True, "teacher": str(teacher)})
