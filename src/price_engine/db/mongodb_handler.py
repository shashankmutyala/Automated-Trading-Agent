from pymongo import MongoClient

class MongoDBHandler:
    def __init__(self, uri="mongodb://localhost:27017/", db_name="price_data"):
        self.client = MongoClient(uri)
        self.db = self.client[db_name]

    def store_price(self, collection_name, symbol, price, timestamp):
        collection = self.db[collection_name]
        data = {
            "symbol": symbol,
            "price": price,
            "timestamp": timestamp,
        }
        collection.insert_one(data)
        print(f"Stored in MongoDB: {data}")
