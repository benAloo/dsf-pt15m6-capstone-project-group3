import getpass
import hashlib
import secrets

ITERATIONS = 200_000

email = input("Admin email: ").strip().lower()
password = getpass.getpass("Admin password: ")
salt = secrets.token_hex(16)


def h(value: str) -> str:
    return hashlib.pbkdf2_hmac("sha256", value.encode(), bytes.fromhex(salt), ITERATIONS).hex()


print("\n# Streamlit secrets")
print(f'admin_salt = "{salt}"')
print(f'admin_email_hash = "{h(email)}"')
print(f'admin_password_hash = "{h(password)}"')
