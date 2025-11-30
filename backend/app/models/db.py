from pymongo import MongoClient
from typing import Optional
import os
from datetime import datetime


class MongoDB:
    _client: Optional[MongoClient] = None
    _db = None

    @classmethod
    def connect(cls):
        if cls._client is None:
            mongo_uri = os.getenv("MONGODB_URI", "mongodb://localhost:27017/")
            cls._client = MongoClient(mongo_uri)
            cls._db = cls._client.get_database("medinsight")
        return cls._db

    @classmethod
    def get_collection(cls, name: str):
        db = cls.connect()
        return db[name]

    @classmethod
    def save_audit(cls, audit_data: dict):
        collection = cls.get_collection("audits")
        audit_data["created_at"] = datetime.utcnow()
        result = collection.insert_one(audit_data)
        return str(result.inserted_id)

    @classmethod
    def get_audit(cls, audit_id: str):
        collection = cls.get_collection("audits")
        from bson import ObjectId
        try:
            return collection.find_one({"_id": ObjectId(audit_id)})
        except:
            return None

    @classmethod
    def close(cls):
        if cls._client:
            cls._client.close()

