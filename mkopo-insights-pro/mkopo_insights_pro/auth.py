from __future__ import annotations

import hashlib
import hmac

ITERATIONS = 200_000
DEFAULT_SALT = "5bf191ae613a96b7ad18db9fb7c1afd8"
DEFAULT_EMAIL_HASH =  "375892fabe6b53d58f47f9292af6e57be8d9ff4cafc0950d3ef92d93dee3ad71"
DEFAULT_PASSWORD_HASH = "e9b19efe24717bc513b7595de672b0cb48370d3dfda5843cbe1237e23673e7e4"


def _hash(value: str, salt_hex: str) -> str:
    return hashlib.pbkdf2_hmac("sha256", value.encode("utf-8"), bytes.fromhex(salt_hex), ITERATIONS).hex()


def _config() -> tuple[str, str, str]:
    """Hashes from Streamlit secrets when present, otherwise the built-in defaults."""
    try:
        import streamlit as st

        if "admin_password_hash" in st.secrets and "admin_email_hash" in st.secrets and "admin_salt" in st.secrets:
            return str(st.secrets["admin_salt"]), str(st.secrets["admin_email_hash"]), str(st.secrets["admin_password_hash"])
    except Exception:
        pass
    return DEFAULT_SALT, DEFAULT_EMAIL_HASH, DEFAULT_PASSWORD_HASH


def verify(email: str, password: str) -> bool:
    salt, email_hash, password_hash = _config()
    ok_email = hmac.compare_digest(_hash((email or "").strip().lower(), salt), email_hash)
    ok_password = hmac.compare_digest(_hash(password or "", salt), password_hash)
    return ok_email and ok_password
