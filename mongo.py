from pymongo.mongo_client import MongoClient


uri = "mongodb+srv://aabidsq:XHITeTqfHF3v3ya3@datacluster.m435l2t.mongodb.net/?retryWrites=true&w=majority"


client = MongoClient(uri)


try:
    client.admin.command("ping")
    print("Pinged your deployment. You successfully connected to MongoDB!")
except Exception as e:
    print(e)
