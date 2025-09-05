from flask import Blueprint, render_template, request, redirect, url_for, flash, jsonify
from db import mongo
from datetime import datetime
from bson import ObjectId
from ai.recognizer import Recognizer
import cv2, numpy as np

# Blueprints
students_bp = Blueprint("students", __name__)  # API
students_ui_bp = Blueprint("students_ui", __name__, template_folder="../templates")  # UI

rec = Recognizer()

# =========================
# API ENDPOINTS
# =========================
@students_bp.get("/")
def list_students():
    """Return all students (omit embeddings)."""
    students = list(mongo.db.students.find({}, {"face_embedding": 0}))
    for s in students:
        s["_id"] = str(s["_id"])
    return jsonify({"ok": True, "students": students})

@students_bp.post("/add")
def add_student_api():
    """Add a student via API (JSON)."""
    data = request.json
    if not data or not all(k in data for k in ["name", "roll_no", "email"]):
        return jsonify({"ok": False, "msg": "Missing required fields"}), 400

    # Check duplicates
    existing = mongo.db.students.find_one({"roll_no": data["roll_no"]})
    if existing:
        return jsonify({"ok": True, "msg": "Student already exists", "student_id": str(existing["_id"])})

    student = {
        "name": data["name"],
        "roll_no": data["roll_no"],
        "email": data["email"],
        "photo_url": data.get("photo_url", ""),
        "face_embedding": data.get("face_embedding", []),
        "created_at": datetime.utcnow()
    }
    result = mongo.db.students.insert_one(student)
    return jsonify({"ok": True, "msg": "Student added", "student_id": str(result.inserted_id)})

# =========================
# WEB UI ENDPOINTS
# =========================
@students_ui_bp.route("/add", methods=["GET", "POST"])
def add_student():
    if request.method == "POST":
        name = request.form.get("name")
        roll_no = request.form.get("roll_no")
        email = request.form.get("email")
        image_file = request.files.get("photo")

        if not image_file:
            flash("Please upload a student photo.", "error")
            return redirect(request.url)

        # Read image and convert to OpenCV format
        img_bytes = image_file.read()
        img_np = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)

        if img is None:
            flash("Failed to read the uploaded image.", "error")
            return redirect(request.url)

        print("Image shape:", img.shape)  # Debug: check if image loaded correctly

        # Detect face and get embeddings
        dets = rec.detect_embed(img)
        print("Detections:", dets)  # Debug: check detection

        if not dets:
            flash("No face detected in the image.", "error")
            return redirect(request.url)

        emb = dets[0][1].astype(float).tolist()
        print("Embedding length:", len(emb))  # Debug: check embedding

        # Insert student in DB
        student = {
            "name": name,
            "roll_no": roll_no,
            "email": email,
            "photo_url": "",  # Optional: store path if needed
            "created_at": datetime.utcnow()
        }
        res = mongo.db.students.insert_one(student)
        user_id = res.inserted_id

        # Save embedding in face_embeddings collection
        mongo.db.face_embeddings.insert_one({
            "user_id": ObjectId(user_id),
            "role": "student",
            "model": "arcface-buffalo_l",
            "embedding": emb,
            "created_at": datetime.utcnow()
        })

        flash(f"Student {name} added and face enrolled successfully!", "success")
        return redirect(url_for("students_ui.add_student"))

    # GET request - render form
    return render_template("add_student.html")


# =========================
# Optionally list students in UI
# =========================
@students_ui_bp.route("/", methods=["GET"])
def list_students_ui():
    students = list(mongo.db.students.find({}, {"face_embedding": 0}))
    for s in students:
        s["_id"] = str(s["_id"])
    return render_template("list_students.html", students=students)
