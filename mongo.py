from pymongo.mongo_client import MongoClient

# It's important to replace "<password>" with the actual password, ensuring special characters are URL-encoded if necessary.
# However, since the provided password doesn't seem to contain special characters that require URL encoding, it can be used directly.
# Note: Directly using passwords in scripts is not a best practice for production code. Consider using environment variables or a secure secret management system.
uri = "mongodb+srv://aabidsq:XHITeTqfHF3v3ya3@datacluster.m435l2t.mongodb.net/?retryWrites=true&w=majority"

# Create a new client and connect to the server
client = MongoClient(uri)

# Send a ping to confirm a successful connection
try:
    client.admin.command("ping")
    print("Pinged your deployment. You successfully connected to MongoDB!")
except Exception as e:
    print(e)
