# backend/main.py
from quart import Quart, request, jsonify, send_from_directory, send_file
from quart_cors import cors
from document_processor import DocumentProcessor
import logging
import os
from dotenv import load_dotenv

# Load environment variables and configure logging
load_dotenv()
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))

# Initialize Quart app with static folder config
app = Quart(__name__, static_folder='static', static_url_path='')
app = cors(app, allow_origin=[
    "http://localhost:5173",
    "http://localhost:50505",
    "http://127.0.0.1:5173",
    "http://127.0.0.1:50505"
])

# Initialize document processor
processor = DocumentProcessor()

# Serve static files - root route and catch all
@app.route('/')
@app.route('/<path:path>')
async def serve(path=''):
    try:
        if path and os.path.exists(os.path.join('static', path)):
            return await send_from_directory('static', path)
        return await send_from_directory('static', 'index.html')
    except Exception as e:
        logging.error(f"Error serving static file: {str(e)}")
        return str(e), 404

@app.route("/api/health")
async def health_check():
    """Health check endpoint"""
    return jsonify({"status": "healthy"})

@app.route("/api/process", methods=["POST"])
async def process_documents():
    """Process documents in the input directory"""
    try:
        stats = processor.process_directory()
        return jsonify({
            "status": "success",
            "stats": stats
        })
    except Exception as e:
        logging.error(f"Error processing documents: {str(e)}")
        return jsonify({
            "status": "error", 
            "message": str(e)
        }), 500

@app.route("/api/query", methods=["POST"])
async def query():
    """Query the document collection"""
    try:
        data = await request.get_json()
        query = data.get("query")
        n_results = data.get("n_results", 3)
        
        if not query:
            return jsonify({"status": "error", "message": "Query is required"}), 400
            
        results = processor.query_documents(query, n_results)
        return jsonify({
            "status": "success",
            "results": results
        })
    except Exception as e:
        logging.error(f"Error querying documents: {str(e)}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route("/api/chat", methods=["POST"])
async def chat():
    """Chat endpoint with RAG"""
    try:
        data = await request.get_json()
        message = data.get("message")
        
        if not message:
            return jsonify({"status": "error", "message": "Message is required"}), 400
        
        docs = processor.query_documents(message, n_results=3)
        context = "\n".join([doc["text"] for doc in docs])
        response = processor.generate_response(message, context)
        
        return jsonify({
            "status": "success",
            "response": response,
            "context": docs
        })
    except Exception as e:
        logging.error(f"Error in chat: {str(e)}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

if __name__ == "__main__":
    port = int(os.getenv("PORT", "50505"))
    app.run(host="0.0.0.0", port=port)