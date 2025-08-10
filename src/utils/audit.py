import sqlite3, os, json, datetime

DB_PATH = "logs/predictions.db"
os.makedirs("logs", exist_ok=True)

def _init():
    with sqlite3.connect(DB_PATH) as con:
        con.execute("""
        CREATE TABLE IF NOT EXISTS predictions(
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          ts TEXT,
          features TEXT,
          prediction INTEGER,
          probabilities TEXT
        )""")
_init()

def log_prediction(features, prediction, probabilities):
    with sqlite3.connect(DB_PATH) as con:
        con.execute(
            "INSERT INTO predictions(ts,features,prediction,probabilities) VALUES (?,?,?,?)",
            (datetime.datetime.utcnow().isoformat(), json.dumps(features), int(prediction), json.dumps(probabilities))
        )