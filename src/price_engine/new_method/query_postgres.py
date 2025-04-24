# new_method/query_postgres.py
import psycopg2
from datetime import datetime
import csv
import sys
import os

# Database connection parameters
conn_params = {
    "host": "postgres",
    "port": 5432,
    "database": "market_data",
    "user": "postgres",
    "password": "password"
}

# File to store the last run timestamp
last_run_file = "last_run.txt"

# Define the symbols and their corresponding CSV files
symbols_to_files = {
    # Binance symbols
    "BTCUSDT": "csv_files/binance_btc.csv",
    "ETHUSDT": "csv_files/binance_eth.csv",
    "SOLUSDT": "csv_files/binance_sol.csv",
    # Uniswap symbols
    "WBTCUSDC": "csv_files/uniswap_wbtc.csv",
    "WETHUSDC": "csv_files/uniswap_weth.csv",
    "SOLUSDC": "csv_files/uniswap_sol.csv"
}

try:
    # Create the csv_files directory if it doesn't exist
    os.makedirs("csv_files", exist_ok=True)

    # Read the last run timestamp
    try:
        with open(last_run_file, "r") as f:
            last_run_timestamp = datetime.fromisoformat(f.read().strip())
        print(f"Last run timestamp: {last_run_timestamp}")
    except FileNotFoundError:
        last_run_timestamp = datetime(2000, 1, 1)
        print(f"No last run timestamp found. Using default: {last_run_timestamp}")

    # Connect to the PostgreSQL database
    conn = psycopg2.connect(**conn_params)
    cursor = conn.cursor()

    # Fetch data for each symbol and save to separate CSV files
    for symbol, output_file in symbols_to_files.items():
        # Query to fetch rows for the specific symbol
        query = """
        SELECT timestamp, symbol, price, volume, source
        FROM prices
        WHERE symbol = %s
          AND timestamp > %s
        ORDER BY timestamp ASC;
        """
        params = (symbol, last_run_timestamp)

        # Get the total number of new rows for this symbol
        cursor.execute("SELECT COUNT(*) FROM prices WHERE symbol = %s AND timestamp > %s;", params)
        total_rows = cursor.fetchone()[0]
        print(f"\nTotal number of new rows for {symbol} since {last_run_timestamp}: {total_rows}")
        sys.stdout.flush()

        if total_rows == 0:
            print(f"No new data for {symbol}. Skipping CSV creation.")
            continue

        # Fetch the new rows
        cursor.execute(query, params)
        
        print(f"\nFetching recent Price Entries for {symbol}:")
        print("Timestamp                  | Symbol    | Price     | Volume | Source")
        print("-" * 70)
        sys.stdout.flush()

        # Open a CSV file to save the data (overwrite mode)
        with open(output_file, mode='w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(["Timestamp", "Symbol", "Price", "Volume", "Source"])

            batch_size = 1000
            rows_processed = 0
            rows_printed = 0
            max_rows_to_print = 10

            while True:
                rows = cursor.fetchmany(batch_size)
                if not rows:
                    break

                for row in rows:
                    timestamp, symbol, price, volume, source = row
                    price = price if price is not None else 0.0
                    volume = volume if volume is not None else 0.0
                    if rows_printed < max_rows_to_print:
                        print(f"{timestamp} | {symbol:9} | {price:9.2f} | {volume:6.2f} | {source}")
                        sys.stdout.flush()
                        rows_printed += 1
                    csv_writer.writerow([timestamp, symbol, price, volume, source])
                    rows_processed += 1

                print(f"Processed {rows_processed}/{total_rows} rows for {symbol}...")
                sys.stdout.flush()

            if rows_printed == max_rows_to_print and total_rows > max_rows_to_print:
                print(f"(Only the first {max_rows_to_print} rows are printed to the console. All {total_rows} rows are saved to {output_file})")
                sys.stdout.flush()

        print(f"All recent data for {symbol} has been saved to {output_file}")
        sys.stdout.flush()

except Exception as e:
    print(f"Error connecting to the database: {e}")
    sys.stdout.flush()

finally:
    if 'cursor' in locals():
        cursor.close()
    if 'conn' in locals():
        conn.close()