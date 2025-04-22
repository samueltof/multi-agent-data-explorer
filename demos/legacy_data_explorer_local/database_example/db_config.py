import os
import sqlite3
import psycopg2

def get_db_connection():
    """Establishes a database connection"""
    if os.getenv("USE_POSTGRES") == "true":
        return psycopg2.connect(
            dbname="roche_data",
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
            host=os.getenv("DB_HOST")
        )
    else:
        return sqlite3.connect("database/roche_data.db")
