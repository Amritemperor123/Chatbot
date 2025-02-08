from flask import Flask, request, jsonify, render_template
from flask_restful import Api, Resource
from sentence_transformers import SentenceTransformer
import numpy as np
from supabase import create_client
import os

# Initialize Flask app and API
app = Flask(__name__, template_folder="templates", static_folder="static")
# app = Flask(__name__)
api = Api(app)

# Load Sentence Transformer model
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Supabase connection
supabase = create_client('https://mpwzvopiompnknvcyuim.supabase.co', 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Im1wd3p2b3Bpb21wbmtudmN5dWltIiwicm9sZSI6ImFub24iLCJpYXQiOjE3Mzg4MjgxOTgsImV4cCI6MjA1NDQwNDE5OH0.dCmdUFgzssB1ZrR9gKckVun57Ss81Q-_blvcmbFDi0Q')

class Chatbot(Resource):
    def post(self):
        try:
            # Get user query
            data = request.get_json()
            query = data.get("query")
            if not query:
                return {"error": "Query is required"}, 400

            # Encode query to get its embedding
            query_embedding = model.encode(query).tolist()

            # Perform similarity search in Supabase
            response = supabase.rpc("match_documents", {"query_embedding": query_embedding, "match_threshold": 0.75, "match_count": 3}).execute()
            matches = response.data

            if not matches or len(matches) == 0:
                return {"response": "Sorry, I couldn't find relevant information."}, 200

            # Return the top match
            return {"response": matches[0]["text"]}, 200

        except Exception as e:
            return {"error": str(e)}, 500


@app.route("/")
def home():
    return render_template("index.html")

# Add API endpoint
api.add_resource(Chatbot, "/chat")

if __name__ == "__main__":
    app.run(debug=True)
