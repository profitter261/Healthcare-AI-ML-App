from flask import Flask, render_template, request, jsonify, redirect, url_for, session, flash
from functools import wraps
import sqlite3, os, pathlib

app = Flask(__name__)
app.secret_key = "supersecretkey"  # Needed for session management
app.config["SESSION_PERMANENT"] = False  # Session ends when browser closes
app.config["SESSION_TYPE"] = "filesystem"  # Ensure server-side sessions

# ---------------- PATHS ----------------
BASE_DIR = pathlib.Path(__file__).parent.resolve()
MAIN_DIR = BASE_DIR.parent
FEEDBACKS_DB = MAIN_DIR / "feedbacks.db"
APPOINTMENTS_DB = os.path.join(BASE_DIR, "appointments.db")

# ---------------- DB INIT ----------------
def init_appointments_db():
    conn = sqlite3.connect(APPOINTMENTS_DB)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS appointments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            date TEXT,
            time TEXT,
            reason TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()
init_appointments_db()

os.makedirs(os.path.dirname(FEEDBACKS_DB), exist_ok=True)
def init_feedbacks_db():
    conn = sqlite3.connect(FEEDBACKS_DB)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS feedbacks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            review TEXT,
            sentiment TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()
init_feedbacks_db()

def get_feedbacks_from_db():
    conn = sqlite3.connect(FEEDBACKS_DB)
    c = conn.cursor()
    c.execute("SELECT id, review, sentiment, created_at FROM feedbacks ORDER BY created_at DESC")
    rows = c.fetchall()
    conn.close()
    return [{"id": r[0], "review": r[1], "sentiment": r[2], "created_at": r[3]} for r in rows]

# ---------------- AUTH DECORATOR ----------------
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not session.get("admin_logged_in"):
            flash("Please login first", "warning")
            return redirect(url_for("admin_login"))
        return f(*args, **kwargs)
    return decorated_function

# No before_request needed - we'll handle auth at route level

# ---------------- ROUTES ----------------
@app.route('/')
def home():
    return render_template('index.html')

# Appointments APIs
@app.route('/api/appointment', methods=['POST'])
def save_appointment():
    data = request.get_json()
    name = data.get("name")
    date = data.get("date")
    time = data.get("time")
    reason = data.get("reason")

    if not (name and date and time and reason):
        return jsonify({"message": "Missing data"}), 400

    conn = sqlite3.connect(APPOINTMENTS_DB)
    c = conn.cursor()
    c.execute("INSERT INTO appointments (name, date, time, reason) VALUES (?, ?, ?, ?)",
              (name, date, time, reason))
    conn.commit()
    conn.close()
    return jsonify({"message": "✅ Appointment saved successfully!"})

@app.route('/api/appointments', methods=['GET'])
def get_appointments():
    conn = sqlite3.connect(APPOINTMENTS_DB)
    c = conn.cursor()
    c.execute("SELECT id, name, date, time, reason, created_at FROM appointments ORDER BY created_at DESC")
    rows = c.fetchall()
    conn.close()
    return jsonify([{"id": r[0], "name": r[1], "date": r[2], "time": r[3], "reason": r[4], "created_at": r[5]} for r in rows])

@app.route('/api/appointment/<int:id>', methods=['DELETE'])
def delete_appointment(id):
    conn = sqlite3.connect(APPOINTMENTS_DB)
    c = conn.cursor()
    c.execute("DELETE FROM appointments WHERE id = ?", (id,))
    conn.commit()
    conn.close()
    return jsonify({"message": f"🗑️ Appointment {id} deleted successfully."})

# Admin login
@app.route('/login', methods=['GET', 'POST'])
def admin_login():
    # If already logged in, redirect to admin panel
    if session.get("admin_logged_in"):
        return redirect(url_for("admin_panel"))

    error = None
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")
        if username == "admin" and password == "admin123":
            session["admin_logged_in"] = True
            flash("✅ Login successful!", "success")
            return redirect(url_for("admin_panel"))
        else:
            error = "❌ Invalid username or password"
            flash(error, "danger")
    
    return render_template("login.html", error=error)

# Admin dashboard (secured)
@app.route('/admin')
@login_required
def admin_panel():
    # Fetch appointments
    conn = sqlite3.connect(APPOINTMENTS_DB)
    c = conn.cursor()
    c.execute("SELECT id, name, date, time, reason, created_at FROM appointments ORDER BY created_at DESC")
    appointments = c.fetchall()
    conn.close()

    feedbacks = get_feedbacks_from_db()

    # Count sentiments
    sentiment_count = {"positive": 0, "neutral": 0, "negative": 0}
    for f in feedbacks:
        if f["sentiment"] in sentiment_count:
            sentiment_count[f["sentiment"]] += 1

    return render_template('admin.html', appointments=appointments, feedbacks=feedbacks, sentiment_count=sentiment_count)

# Admin logout
@app.route('/admin/logout')
def admin_logout():
    session.pop("admin_logged_in", None)
    flash("✅ Logged out successfully!", "info")
    return redirect(url_for("admin_login"))

# ---------------- RUN ----------------
if __name__ == "__main__":
    app.run(debug=True)