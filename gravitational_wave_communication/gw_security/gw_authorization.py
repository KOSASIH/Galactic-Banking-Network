import json
import jwt
import datetime
from datetime import timedelta
from flask import Flask, request, jsonify
from flask_jwt_extended import JWTManager, jwt_required, create_access_token, get_jwt_identity
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String, DateTime, Boolean
from sqlalchemy.dialects.postgresql import JSONB

app = Flask(__name__)
app.config['JWT_SECRET_KEY'] = 'super-secret-key'
app.config['JWT_ACCESS_TOKEN_EXPIRES'] = timedelta(hours=1)
jwt = JWTManager(app)
CORS(app)

db = SQLAlchemy(app)
Base = declarative_base()

class User(Base):
    __tablename__ = 'users'
    id = Column(Integer, primary_key=True)
    username = Column(String(50), unique=True, nullable=False)
    password = Column(String(100), nullable=False)
    roles = Column(JSONB, nullable=False, default=[])

    def __init__(self, username, password, roles):
        self.username = username
        self.password = password
        self.roles = roles

    def __repr__(self):
        return f"User ('{self.username}', '{self.password}', {self.roles})"

@app.route('/login', methods=['POST'])
def login():
    username = request.json.get('username')
    password = request.json.get('password')
    user = User.query.filter_by(username=username).first()
    if user and user.password == password:
        access_token = create_access_token(identity=username)
        return jsonify(access_token=access_token)
    return jsonify({"msg": "Bad username or password"}), 401

@app.route('/protected', methods=['GET'])
@jwt_required
def protected():
    username = get_jwt_identity()
    user = User.query.filter_by(username=username).first()
    if user:
        return jsonify({"username": username, "roles": user.roles})
    return jsonify({"msg": "User  not found"}), 404

@app.route('/authorize', methods=['POST'])
@jwt_required
def authorize():
    username = get_jwt_identity()
    user = User.query.filter_by(username=username).first()
    if user:
        roles = user.roles
        if 'admin' in roles:
            return jsonify({"authorized": True})
        else:
            return jsonify({"authorized": False})
    return jsonify({"msg": "User  not found"}), 404

if __name__ == '__main__':
    app.run(debug=True)
