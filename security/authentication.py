import jwt

def generate_token(user_id):
    secret_key = secrets.get("secret_key")
    token = jwt.encode({"user_id": user_id}, secret_key, algorithm="HS256")
    return token

def verify_token(token):
    secret_key = secrets.get("secret_key")
    try:
        payload = jwt.decode(token, secret_key, algorithms=["HS256"])
        return payload["user_id"]
    except jwt.ExpiredSignatureError:
        return None
    except jwt.InvalidTokenError:
        return None
