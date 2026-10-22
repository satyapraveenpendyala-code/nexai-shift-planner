"""
db.py - SQLite storage for users (hashed passwords) and the manager's shift settings.
The manager's settings are stored here so a technician logging in separately sees the same plan.
"""
import hashlib
import json
import os
import sqlite3
from pathlib import Path

DB = Path(__file__).parent / "planner.db"
MAX_TECH = 20
_SKILLS = ["Electrical", "Mechanical", "General"]  # roster repeats this skill pattern
TECH_SLOTS = [f"Tech {i+1} ({_SKILLS[i % 3]})" for i in range(MAX_TECH)]
DEFAULT_SETTINGS = {"n_tech": 3, "shift_size": 100, "shift_no": 1, "ratio": 20,
                    "pwf": 25, "osf": 20, "hdf": 12, "twf": 10}
DEMO_USERS = [("manager", "manager123", "Manager", None),
              ("tech1", "tech123", "Technician", TECH_SLOTS[0]),
              ("tech2", "tech123", "Technician", TECH_SLOTS[1]),
              ("tech3", "tech123", "Technician", TECH_SLOTS[2])]


def _conn():
    return sqlite3.connect(DB, check_same_thread=False)


def _hash(password, salt):
    return hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 100_000).hex()


def init():
    with _conn() as c:
        c.execute("""CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY, salt TEXT, pw_hash TEXT, role TEXT, tech_slot TEXT)""")
        c.execute("CREATE TABLE IF NOT EXISTS settings (id INTEGER PRIMARY KEY CHECK (id = 1), data TEXT)")
        c.execute("INSERT OR IGNORE INTO settings VALUES (1, ?)", (json.dumps(DEFAULT_SETTINGS),))
    for u in DEMO_USERS:
        create_user(*u)


def create_user(username, password, role, tech_slot=None):
    """Returns (ok, message)."""
    username = username.strip().lower()
    if len(username) < 3 or len(password) < 6:
        return False, "Username needs 3+ characters and password 6+ characters."
    salt = os.urandom(16)
    try:
        with _conn() as c:
            c.execute("INSERT INTO users VALUES (?, ?, ?, ?, ?)",
                      (username, salt.hex(), _hash(password, salt), role, tech_slot))
        return True, "Account created. You can log in now."
    except sqlite3.IntegrityError:
        return False, "That username is already taken."


def verify(username, password, role):
    """Returns the user dict if the username, password and profile match, else None."""
    with _conn() as c:
        row = c.execute("SELECT salt, pw_hash, role, tech_slot FROM users WHERE username = ?",
                        (username.strip().lower(),)).fetchone()
    if not row or row[2] != role or _hash(password, bytes.fromhex(row[0])) != row[1]:
        return None
    return {"username": username.strip().lower(), "role": row[2], "tech_slot": row[3]}


def get_settings():
    with _conn() as c:
        return {**DEFAULT_SETTINGS, **json.loads(c.execute("SELECT data FROM settings").fetchone()[0])}


def save_settings(s):
    with _conn() as c:
        c.execute("UPDATE settings SET data = ? WHERE id = 1", (json.dumps(s),))