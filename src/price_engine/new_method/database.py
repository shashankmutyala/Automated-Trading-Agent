# new_method/database.py
import psycopg2
import logging

logger = logging.getLogger(__name__)

class Database:
    def __init__(self):
        self.conn = psycopg2.connect(
            host="postgres",
            port=5432,
            database="market_data",
            user="postgres",
            password="password"
        )
        self.create_table()

    def create_table(self):
        with self.conn.cursor() as cursor:
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS prices (
                    timestamp TIMESTAMP,
                    symbol VARCHAR(20),
                    price FLOAT,
                    volume FLOAT,
                    source VARCHAR(50)
                );
            """)
            self.conn.commit()

    def store_price(self, data, symbol):
        with self.conn.cursor() as cursor:
            cursor.execute("""
                INSERT INTO prices (timestamp, symbol, price, volume, source)
                VALUES (%s, %s, %s, %s, %s);
            """, (data["timestamp"], symbol, data["price"], data.get("volume", 0.0), data["source"]))
            self.conn.commit()

    def close(self):
        self.conn.close()