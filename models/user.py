import bcrypt

class User:
    def __init__(self, username, password, email):
        self.username = username
        self.password_hash = self._hash_password(password)
        self.email = email

    def _hash_password(self, password):
        salt = bcrypt.gensalt()
        password_hash = bcrypt.hashpw(password.encode(), salt)
        return password_hash

    def authenticate(self, password):
        return bcrypt.checkpw(password.encode(), self.password_hash)
