# app.py - Flask API wrapper for Legal RAG Pipeline (Production Version)

import os
import logging
from datetime import datetime
from typing import List
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import tempfile
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import your main pipeline
from main_pipeline import LegalRAGPipeline

# ------------------------
# Logging
# ------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ------------------------
# Flask app & CORS
# ------------------------
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)  # Allow all origins for testing

app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max

# ------------------------
# Pipeline initialization
# ------------------------
pipeline = None

def init_pipeline():
    global pipeline
    try:
        pipeline = LegalRAGPipeline()
        logger.info("Pipeline initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize pipeline: {e}")
        return False

if not init_pipeline():
    logger.error("Pipeline failed to initialize!")

# ------------------------
# Utility functions
# ------------------------
ALLOWED_EXTENSIONS = {'pdf', 'docx', 'doc', 'txt', 'rtf', 'odt'}
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def validate_file_size(file):
    file.seek(0, 2)  # end
    size = file.tell()
    file.seek(0)
    return size <= MAX_FILE_SIZE

def handle_error(msg, code=500):
    logger.error(msg)
    return jsonify({"success": False, "error": msg, "timestamp": datetime.now().isoformat()}), code

def validate_json_request(required_fields: List[str] = None):
    if not request.is_json:
        return False, "Request must be JSON"
    data = request.get_json()
    if not data:
        return False, "Invalid JSON data"
    if required_fields:
        missing = [f for f in required_fields if f not in data]
        if missing:
            return False, f"Missing required fields: {missing}"
    return True, data

# ------------------------
# Health & Status
# ------------------------
@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy",
        "pipeline_ready": pipeline is not None and pipeline.pipeline_ready,
        "timestamp": datetime.now().isoformat()
    })

@app.route("/status", methods=["GET"])
def status():
    try:
        if not pipeline:
            return handle_error("Pipeline not initialized", 503)
        return jsonify({"success": True, "status": pipeline.get_enhanced_pipeline_status(), "timestamp": datetime.now().isoformat()})
    except Exception as e:
        return handle_error(str(e))

# ------------------------
# Upload & Process
# ------------------------
@app.route("/upload", methods=["POST"])
def upload_files():
    try:
        if 'files' not in request.files:
            return handle_error("No files provided", 400)
        files = request.files.getlist('files')
        if not files:
            return handle_error("No files selected", 400)

        uploaded_paths = []
        warnings = []

        for f in files:
            if f and f.filename:
                if not allowed_file(f.filename):
                    warnings.append(f"File type not allowed: {f.filename}")
                    continue
                if not validate_file_size(f):
                    warnings.append(f"File too large: {f.filename}")
                    continue
                tmp = tempfile.NamedTemporaryFile(delete=False)
                f.save(tmp.name)
                uploaded_paths.append(tmp.name)

        if not uploaded_paths and warnings:
            return handle_error("Upload failed: " + "; ".join(warnings), 400)

        result = {"success": True, "files_uploaded": len(uploaded_paths), "file_paths": uploaded_paths}
        if warnings:
            result["warnings"] = warnings
        return jsonify(result)
    except Exception as e:
        return handle_error(str(e))

@app.route("/process", methods=["POST"])
def process_documents():
    try:
        valid, data = validate_json_request(['file_paths'])
        if not valid:
            return handle_error(data, 400)

        file_paths = [fp for fp in data['file_paths'] if fp and isinstance(fp, str)]
        if not file_paths:
            return handle_error("No valid file paths provided", 400)

        missing_files = [fp for fp in file_paths if not os.path.exists(fp)]
        if missing_files:
            return handle_error(f"Files not found: {missing_files}", 404)

        store_prefix = data.get("store_prefix")
        result = pipeline.process_new_documents_with_categories(file_paths, store_prefix)
        return jsonify({"success": True, "result": result, "timestamp": datetime.now().isoformat()})
    except Exception as e:
        return handle_error(str(e))

# ------------------------
# Query endpoints example
# ------------------------
@app.route("/query", methods=["POST"])
def query_documents():
    try:
        valid, data = validate_json_request(['question'])
        if not valid:
            return handle_error(data, 400)
        question = data['question']
        category = data.get("category")
        result = pipeline.query_documents(question, category)
        return jsonify({"success": True, "result": result, "timestamp": datetime.now().isoformat()})
    except Exception as e:
        return handle_error(str(e))

# ------------------------
# Add all other endpoints similarly (load_stores, compare, summary, etc.)
# ------------------------
# You can copy your existing route functions but remove any reference to local log files
# and ensure any temporary files are handled within the request.

# ------------------------
# Production-ready entrypoint
# ------------------------
# Vercel handles the WSGI function automatically. No need for app.run()

# This makes the app compatible with WSGI servers
application = app