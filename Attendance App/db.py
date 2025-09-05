from flask_pymongo import PyMongo
from flask import Flask

mongo = PyMongo()

def init_db(app: Flask):
  app.config["MONGO_URI"] = "mongodb://localhost:27017/smart_attendance"
  mongo.init_app(app)
  return mongo
