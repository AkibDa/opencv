from flask import Flask, jsonify
from db import init_db
from routes.students import students_bp, students_ui_bp
from routes.teachers import teachers_bp
from routes.enroll import enroll_bp
from routes.sessions import sessions_bp
from routes.attendance import attendance_bp
import os

app = Flask(__name__)
app.secret_key = "your_super_secret_key_here"
mongo = init_db(app)

# Register blueprints with clean URL prefixes
app.register_blueprint(students_bp, url_prefix="/api/students")
app.register_blueprint(students_ui_bp , url_prefix="/students")
app.register_blueprint(teachers_bp, url_prefix="/api/teachers")
app.register_blueprint(enroll_bp, url_prefix="/api/enroll")
app.register_blueprint(sessions_bp, url_prefix="/api/sessions")
app.register_blueprint(attendance_bp, url_prefix="/api/attendance")

@app.route("/", methods=["GET"])
def index():
    return jsonify({
        "ok": True,
        "msg": "Smart Attendance API is running",
        "endpoints": {
            "students": "/api/students",
            "teachers": "/api/teachers",
            "enroll_face": "/api/enroll/face",
            "sessions": "/api/sessions",
            "attendance_recognize": "/api/attendance/recognize"
        }
    })

@app.route("/api/test")
def test_api():
    return {"ok": True, "msg": "API working"}


# Error handlers
@app.errorhandler(404)
def not_found(e):
    return jsonify({"ok": False, "msg": "Endpoint not found"}), 404

@app.errorhandler(500)
def server_error(e):
    return jsonify({"ok": False, "msg": "Internal server error"}), 500

if __name__ == "__main__":
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 3000))
    debug = os.getenv("DEBUG", "True") == "True"
    app.run(host=host, port=port, debug=debug)
