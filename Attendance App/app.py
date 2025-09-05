from flask import Flask
from db import init_db
from routes.students import students_bp
from routes.teachers import teachers_bp
from routes.enroll import enroll_bp
from routes.sessions import sessions_bp
from routes.attendance import attendance_bp

app = Flask(__name__)
mongo = init_db(app)

# Register blueprints
app.register_blueprint(students_bp)
app.register_blueprint(teachers_bp)
app.register_blueprint(enroll_bp)
app.register_blueprint(sessions_bp)
app.register_blueprint(attendance_bp)

if __name__ == "__main__":
    app.run(debug=True)
