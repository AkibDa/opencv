from flask_pymongo import PyMongo
from flask import Flask
import os

mongo = PyMongo()

def init_db(app: Flask):
  # Get Mongo URI from environment variable or use default
  mongo_uri = os.getenv("MONGO_URI", "mongodb://localhost:27017/smart_attendance")
  app.config["MONGO_URI"] = mongo_uri

  try:
    mongo.init_app(app)
    print(f"[INFO] Connected to MongoDB at {mongo_uri}")
  except Exception as e:
    print(f"[ERROR] Could not connect to MongoDB: {e}")
    raise e

  return mongo
