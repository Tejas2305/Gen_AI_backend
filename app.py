# app.py - Flask API wrapper for Legal RAG Pipeline (Production Version)

import os
import logging
from datetime import datetime
from typing import List
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename
from werkzeug.exceptions import RequestEntityTooLarge
import tempfile
import shutil
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import your main pipeline
from main_pipeline import LegalRAGPipeline

# Setup logging for production
def setup_logging():
    """Setup logging based on environment"""
    log_level = logging.INFO
    
    # In production, only log to console (Render captures this)
    if os.environ.get('RENDER') or os.environ.get('FLASK_ENV') == 'production':
        handlers = [logging.StreamHandler()]
    else:
        # In development, log to both file and console
        os.makedirs('logs', exist_ok=True)
        handlers = [
            logging.FileHandler('logs/flask_api.log'),
            logging.StreamHandler()
        ]
    
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )

setup_logging()
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size

# --- FIXED CORS SETUP BELOW ---
def setup_cors():
    """
    Setup CORS to allow localhost:5173 (dev) and production frontend.
    You can add more origins as needed.
    """
    allowed_origins = [
        "http://localhost:5173",
        "https://legalai-7737e.web.app",
        # Add your production frontend URL here if different
    ]
    # For quick development, you may use '*' (not for production!)
    CORS(app, resources={r"/*": {"origins": allowed_origins}}, supports_credentials=True)
    logger.info(f"CORS configured - Origins: {allowed_origins}")

setup_cors()

# Global pipeline instance
pipeline = None
temp_upload_dir = None

# File validation settings
ALLOWED_EXTENSIONS = {'pdf', 'docx', 'doc', 'txt', 'rtf', 'odt'}
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB per file

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def validate_file_size(file):
    """Validate individual file size"""
    file.seek(0, 2)  # Seek to end
    file_size = file.tell()
    file.seek(0)  # Reset to beginning
    return file_size <= MAX_FILE_SIZE

def init_pipeline():
    """Initialize the pipeline singleton"""
    global pipeline, temp_upload_dir
    try:
        pipeline = LegalRAGPipeline()
        # Create temp directory with better error handling
        temp_upload_dir = tempfile.mkdtemp(prefix="legal_rag_")
        
        # Ensure directory exists and has proper permissions
        if not os.path.exists(temp_upload_dir):
            os.makedirs(temp_upload_dir, exist_ok=True)
        
        logger.info(f"Pipeline initialized successfully. Temp dir: {temp_upload_dir}")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize pipeline: {e}")
        return False

def cleanup_temp_files():
    """Clean up temporary upload directory"""
    global temp_upload_dir
    if temp_upload_dir and os.path.exists(temp_upload_dir):
        try:
            shutil.rmtree(temp_upload_dir)
            logger.info("Temporary files cleaned up")
        except Exception as e:
            logger.error(f"Error cleaning temp files: {e}")

def handle_error(error_message: str, status_code: int = 500):
    """Standardized error response"""
    logger.error(error_message)
    return jsonify({
        "success": False,
        "error": error_message,
        "timestamp": datetime.now().isoformat()
    }), status_code

def validate_json_request(required_fields: List[str] = None):
    """Validate JSON request and required fields"""
    if not request.is_json:
        return False, "Request must be JSON"
    
    data = request.get_json()
    if not data:
        return False, "Invalid JSON data"
    
    if required_fields:
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            return False, f"Missing required fields: {missing_fields}"
    
    return True, data

# ============================================================================
# HEALTH CHECK & STATUS ENDPOINTS
# ============================================================================

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "service": "Legal RAG Pipeline API",
        "timestamp": datetime.now().isoformat(),
        "pipeline_ready": pipeline is not None and pipeline.pipeline_ready,
        "environment": os.environ.get('FLASK_ENV', 'development')
    })

@app.route('/status', methods=['GET'])
def get_status():
    """Get comprehensive pipeline status"""
    try:
        if not pipeline:
            return handle_error("Pipeline not initialized", 503)
        
        status = pipeline.get_enhanced_pipeline_status()
        return jsonify({
            "success": True,
            "status": status,
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        return handle_error(f"Error getting status: {str(e)}")

# ============================================================================
# TESTING ENDPOINTS
# ============================================================================

@app.route('/test_paths', methods=['POST'])
def test_file_paths():
    """Test endpoint to debug file path issues"""
    try:
        data = request.get_json()
        
        return jsonify({
            "success": True,
            "received_data": data,
            "file_paths": data.get('file_paths') if data else None,
            "file_paths_type": str(type(data.get('file_paths'))) if data else None,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        return handle_error(f"Test error: {str(e)}")

# ============================================================================
# DOCUMENT PROCESSING ENDPOINTS
# ============================================================================

@app.route('/upload', methods=['POST'])
def upload_files():
    """Upload files for processing with validation"""
    try:
        if not temp_upload_dir:
            return handle_error("Upload directory not initialized", 503)
        
        # Ensure temp directory still exists
        if not os.path.exists(temp_upload_dir):
            os.makedirs(temp_upload_dir, exist_ok=True)
            logger.info(f"Recreated temp directory: {temp_upload_dir}")
        
        if 'files' not in request.files:
            return handle_error("No files provided", 400)
        
        files = request.files.getlist('files')
        if not files or all(file.filename == '' for file in files):
            return handle_error("No files selected", 400)
        
        # Validate and save uploaded files
        uploaded_files = []
        errors = []
        
        for file in files:
            if file and file.filename:
                # Check file extension
                if not allowed_file(file.filename):
                    errors.append(f"File type not allowed: {file.filename}")
                    continue
                
                # Check file size
                if not validate_file_size(file):
                    errors.append(f"File too large: {file.filename} (max 50MB)")
                    continue
                
                # Save file
                filename = secure_filename(file.filename)
                # Add timestamp to avoid conflicts
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{timestamp}_{filename}"
                file_path = os.path.join(temp_upload_dir, filename)
                
                try:
                    # Double-check directory exists before saving
                    os.makedirs(os.path.dirname(file_path), exist_ok=True)
                    file.save(file_path)
                    
                    # Verify file was actually saved
                    if os.path.exists(file_path):
                        uploaded_files.append(file_path)
                        logger.info(f"Successfully saved file: {file_path}")
                    else:
                        errors.append(f"File save verification failed: {file.filename}")
                        
                except Exception as e:
                    logger.error(f"Failed to save {file.filename}: {str(e)}")
                    errors.append(f"Failed to save {file.filename}: {str(e)}")
        
        if not uploaded_files and errors:
            return handle_error(f"Upload failed: {'; '.join(errors)}", 400)
        
        result = {
            "success": True,
            "files_uploaded": len(uploaded_files),
            "file_paths": uploaded_files,
            "message": "Files uploaded successfully. Use /process endpoint to process them.",
            "timestamp": datetime.now().isoformat()
        }
        
        if errors:
            result["warnings"] = errors
        
        return jsonify(result)
        
    except RequestEntityTooLarge:
        return handle_error("File too large (max 100MB total)", 413)
    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        return handle_error(f"Upload error: {str(e)}")

@app.route('/process', methods=['POST'])
def process_documents():
    """Process documents with categorization"""
    try:
        if not pipeline:
            return handle_error("Pipeline not initialized", 503)
        
        valid, data = validate_json_request(['file_paths'])
        if not valid:
            return handle_error(data, 400)
        
        file_paths = data['file_paths']
        
        # Fix: Handle None values and filter them out
        if not file_paths:
            return handle_error("No file paths provided", 400)
        
        # Filter out None/null values and ensure all are strings
        valid_file_paths = []
        for fp in file_paths:
            if fp is not None and isinstance(fp, str) and fp.strip():
                valid_file_paths.append(fp.strip())
        
        if not valid_file_paths:
            return handle_error("No valid file paths provided", 400)
        
        store_prefix = data.get('store_prefix')
        
        # Validate file paths exist
        missing_files = [fp for fp in valid_file_paths if not os.path.exists(fp)]
        if missing_files:
            return handle_error(f"Files not found: {missing_files}", 404)
        
        result = pipeline.process_new_documents_with_categories(valid_file_paths, store_prefix)
        
        return jsonify({
            "success": True,
            "result": result,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        return handle_error(f"Processing error: {str(e)}")

@app.route('/load_stores', methods=['POST'])
def load_existing_stores():
    """Load existing category stores"""
    try:
        if not pipeline:
            return handle_error("Pipeline not initialized", 503)
        
        valid, data = validate_json_request(['store_prefix'])
        if not valid:
            return handle_error(data, 400)
        
        store_prefix = data['store_prefix']
        result = pipeline.load_existing_category_stores(store_prefix)
        
        return jsonify({
            "success": True,
            "result": result,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        return handle_error(f"Load stores error: {str(e)}")

# ============================================================================
# QUERY ENDPOINTS
# ============================================================================

@app.route('/query', methods=['POST'])
def query_documents():
    """Query documents (all categories or specific category)"""
    try:
        if not pipeline:
            return handle_error("Pipeline not initialized", 503)
        
        if not pipeline.pipeline_ready:
            return handle_error("Pipeline not ready. Process documents first.", 400)
        
        valid, data = validate_json_request(['question'])
        if not valid:
            return handle_error(data, 400)
        
        question = data['question']
        category = data.get('category')
        
        result = pipeline.query_documents(question, category)
        
        return jsonify({
            "success": True,
            "result": result,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        return handle_error(f"Query error: {str(e)}")

@app.route('/query_category/<category>', methods=['POST'])
def query_specific_category(category):
    """Query documents within a specific category"""
    try:
        if not pipeline:
            return handle_error("Pipeline not initialized", 503)
        
        valid, data = validate_json_request(['question'])
        if not valid:
            return handle_error(data, 400)
        
        question = data['question']
        result = pipeline.query_category(question, category)
        
        return jsonify({
            "success": True,
            "result": result,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        return handle_error(f"Category query error: {str(e)}")

@app.route('/query_file', methods=['POST'])
def query_by_file():
    """Query a specific file directly"""
    try:
        if not pipeline:
            return handle_error("Pipeline not initialized", 503)
        
        valid, data = validate_json_request(['question', 'file_path'])
        if not valid:
            return handle_error(data, 400)
        
        question = data['question']
        file_path = data['file_path']
        
        if not os.path.exists(file_path):
            return handle_error("File not found", 404)
        
        result = pipeline.query_documents_by_file(question, file_path)
        
        return jsonify({
            "success": True,
            "result": result,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        return handle_error(f"File query error: {str(e)}")

# ============================================================================
# COMPARISON ENDPOINTS
# ============================================================================

@app.route('/compare', methods=['POST'])
def compare_documents():
    """Compare documents between two categories"""
    try:
        if not pipeline:
            return handle_error("Pipeline not initialized", 503)
        
        valid, data = validate_json_request(['question', 'category1', 'category2'])
        if not valid:
            return handle_error(data, 400)
        
        question = data['question']
        category1 = data['category1']
        category2 = data['category2']
        
        result = pipeline.compare_documents(question, category1, category2)
        
        return jsonify({
            "success": True,
            "result": result,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        return handle_error(f"Comparison error: {str(e)}")

@app.route('/compare_files', methods=['POST'])
def compare_files():
    """Compare two specific files"""
    try:
        if not pipeline:
            return handle_error("Pipeline not initialized", 503)
        
        valid, data = validate_json_request(['question', 'file_path1', 'file_path2'])
        if not valid:
            return handle_error(data, 400)
        
        question = data['question']
        file_path1 = data['file_path1']
        file_path2 = data['file_path2']
        
        # Validate files exist
        for fp in [file_path1, file_path2]:
            if not os.path.exists(fp):
                return handle_error(f"File not found: {fp}", 404)
        
        result = pipeline.compare_documents_by_file(question, file_path1, file_path2)
        
        return jsonify({
            "success": True,
            "result": result,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        return handle_error(f"File comparison error: {str(e)}")

@app.route('/compare_obligations', methods=['POST'])
def compare_obligations():
    """Compare obligations between two categories"""
    try:
        if not pipeline:
            return handle_error("Pipeline not initialized", 503)
        
        valid, data = validate_json_request(['category1', 'category2'])
        if not valid:
            return handle_error(data, 400)
        
        result = pipeline.compare_obligations(data['category1'], data['category2'])
        
        return jsonify({
            "success": True,
            "result": result,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        return handle_error(f"Obligations comparison error: {str(e)}")

@app.route('/compare_termination', methods=['POST'])
def compare_termination():
    """Compare termination clauses between two categories"""
    try:
        if not pipeline:
            return handle_error("Pipeline not initialized", 503)
        
        valid, data = validate_json_request(['category1', 'category2'])
        if not valid:
            return handle_error(data, 400)
        
        result = pipeline.compare_termination_clauses(data['category1'], data['category2'])
        
        return jsonify({
            "success": True,
            "result": result,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        return handle_error(f"Termination comparison error: {str(e)}")

@app.route('/compare_clauses', methods=['POST'])
def compare_specific_clauses():
    """Compare specific clauses between two categories"""
    try:
        if not pipeline:
            return handle_error("Pipeline not initialized", 503)
        
        valid, data = validate_json_request(['clause_description', 'category1', 'category2'])
        if not valid:
            return handle_error(data, 400)
        
        result = pipeline.compare_specific_clauses(
            data['clause_description'], 
            data['category1'], 
            data['category2']
        )
        
        return jsonify({
            "success": True,
            "result": result,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        return handle_error(f"Clause comparison error: {str(e)}")

# ============================================================================
# ANALYSIS ENDPOINTS
# ============================================================================

@app.route('/summary', methods=['POST'])
def get_summary():
    """Get document summary (all or specific category)"""
    try:
        if not pipeline:
            return handle_error("Pipeline not initialized", 503)
        
        data = request.get_json() or {}
        category = data.get('category')
        
        result = pipeline.get_document_summary(category)
        
        return jsonify({
            "success": True,
            "result": result,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        return handle_error(f"Summary error: {str(e)}")

@app.route('/summary_file', methods=['POST'])
def get_file_summary():
    """Get summary of a specific file"""
    try:
        if not pipeline:
            return handle_error("Pipeline not initialized", 503)
        
        valid, data = validate_json_request(['file_path'])
        if not valid:
            return handle_error(data, 400)
        
        file_path = data['file_path']
        if not os.path.exists(file_path):
            return handle_error("File not found", 404)
        
        result = pipeline.get_document_summary_by_file(file_path)
        
        return jsonify({
            "success": True,
            "result": result,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        return handle_error(f"File summary error: {str(e)}")

@app.route('/obligations', methods=['POST'])
def find_obligations():
    """Find key obligations (all or specific category)"""
    try:
        if not pipeline:
            return handle_error("Pipeline not initialized", 503)
        
        data = request.get_json() or {}
        category = data.get('category')
        
        result = pipeline.find_key_obligations(category)
        
        return jsonify({
            "success": True,
            "result": result,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        return handle_error(f"Obligations analysis error: {str(e)}")

@app.route('/obligations_file', methods=['POST'])
def find_obligations_by_file():
    """Find key obligations in a specific file"""
    try:
        if not pipeline:
            return handle_error("Pipeline not initialized", 503)
        
        valid, data = validate_json_request(['file_path'])
        if not valid:
            return handle_error(data, 400)
        
        file_path = data['file_path']
        if not os.path.exists(file_path):
            return handle_error("File not found", 404)
        
        result = pipeline.find_obligations_by_file(file_path)
        
        return jsonify({
            "success": True,
            "result": result,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        return handle_error(f"File obligations error: {str(e)}")

@app.route('/termination', methods=['POST'])
def find_termination():
    """Find termination clauses (all or specific category)"""
    try:
        if not pipeline:
            return handle_error("Pipeline not initialized", 503)
        
        data = request.get_json() or {}
        category = data.get('category')
        
        result = pipeline.find_termination_clauses(category)
        
        return jsonify({
            "success": True,
            "result": result,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        return handle_error(f"Termination analysis error: {str(e)}")

@app.route('/termination_file', methods=['POST'])
def find_termination_by_file():
    """Find termination clauses in a specific file"""
    try:
        if not pipeline:
            return handle_error("Pipeline not initialized", 503)
        
        valid, data = validate_json_request(['file_path'])
        if not valid:
            return handle_error(data, 400)
        
        file_path = data['file_path']
        if not os.path.exists(file_path):
            return handle_error("File not found", 404)
        
        result = pipeline.find_termination_clauses_by_file(file_path)
        
        return jsonify({
            "success": True,
            "result": result,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        return handle_error(f"File termination error: {str(e)}")

@app.route('/explain_clause', methods=['POST'])
def explain_clause():
    """Explain a specific clause"""
    try:
        if not pipeline:
            return handle_error("Pipeline not initialized", 503)
        
        valid, data = validate_json_request(['clause_description'])
        if not valid:
            return handle_error(data, 400)
        
        clause_description = data['clause_description']
        category = data.get('category')
        
        result = pipeline.explain_specific_clause(clause_description, category)
        
        return jsonify({
            "success": True,
            "result": result,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        return handle_error(f"Clause explanation error: {str(e)}")

@app.route('/explain_clause_file', methods=['POST'])
def explain_clause_by_file():
    """Explain a specific clause in a file"""
    try:
        if not pipeline:
            return handle_error("Pipeline not initialized", 503)
        
        valid, data = validate_json_request(['clause_description', 'file_path'])
        if not valid:
            return handle_error(data, 400)
        
        clause_description = data['clause_description']
        file_path = data['file_path']
        
        if not os.path.exists(file_path):
            return handle_error("File not found", 404)
        
        result = pipeline.explain_clause_by_file(clause_description, file_path)
        
        return jsonify({
            "success": True,
            "result": result,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        return handle_error(f"File clause explanation error: {str(e)}")

# ============================================================================
# CATEGORY & METADATA ENDPOINTS
# ============================================================================

@app.route('/categories', methods=['GET'])
def get_categories():
    """Get available categories"""
    try:
        if not pipeline:
            return handle_error("Pipeline not initialized", 503)
        
        categories = pipeline.get_available_categories()
        category_info = pipeline.get_category_info() if pipeline.pipeline_ready else {}
        
        return jsonify({
            "success": True,
            "categories": categories,
            "category_info": category_info,
            "total_categories": len(categories),
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        return handle_error(f"Categories error: {str(e)}")

@app.route('/categorizations', methods=['GET'])
def get_categorizations():
    """Get categorization results"""
    try:
        if not pipeline:
            return handle_error("Pipeline not initialized", 503)
        
        categorizations = pipeline.get_categorizations()
        
        return jsonify({
            "success": True,
            "categorizations": categorizations,
            "count": len(categorizations),
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        return handle_error(f"Categorizations error: {str(e)}")

@app.route('/export_report', methods=['POST'])
def export_categorization_report():
    """Export categorization report"""
    try:
        if not pipeline:
            return handle_error("Pipeline not initialized", 503)
        
        data = request.get_json() or {}
        filepath = data.get('filepath')
        
        report_path = pipeline.export_categorization_report(filepath)
        
        # Option to return file directly or just the path
        if data.get('download', False):
            return send_file(report_path, as_attachment=True)
        else:
            return jsonify({
                "success": True,
                "report_path": report_path,
                "timestamp": datetime.now().isoformat()
            })
        
    except Exception as e:
        return handle_error(f"Export error: {str(e)}")

# ============================================================================
# CONVERSATION MANAGEMENT ENDPOINTS
# ============================================================================

@app.route('/conversation/clear', methods=['POST'])
def clear_conversation():
    """Clear conversation history"""
    try:
        if not pipeline:
            return handle_error("Pipeline not initialized", 503)
        
        pipeline.clear_conversation()
        
        return jsonify({
            "success": True,
            "message": "Conversation history cleared",
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        return handle_error(f"Clear conversation error: {str(e)}")

@app.route('/conversation/history', methods=['GET'])
def get_conversation_history():
    """Get conversation history"""
    try:
        if not pipeline:
            return handle_error("Pipeline not initialized", 503)
        
        history = pipeline.get_conversation_history()
        
        return jsonify({
            "success": True,
            "history": history,
            "message_count": len(history),
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        return handle_error(f"Get conversation history error: {str(e)}")

# ============================================================================
# STORE MANAGEMENT ENDPOINTS
# ============================================================================

@app.route('/stores/delete', methods=['DELETE'])
def delete_stores():
    """Delete category stores"""
    try:
        if not pipeline:
            return handle_error("Pipeline not initialized", 503)
        
        data = request.get_json() or {}
        store_prefix = data.get('store_prefix')
        
        results = pipeline.delete_category_stores(store_prefix)
        
        return jsonify({
            "success": True,
            "deletion_results": results,
            "stores_deleted": sum(results.values()),
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        return handle_error(f"Delete stores error: {str(e)}")

# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.errorhandler(404)
def not_found(error):
    return jsonify({
        "success": False,
        "error": "Endpoint not found",
        "timestamp": datetime.now().isoformat()
    }), 404

@app.errorhandler(405)
def method_not_allowed(error):
    return jsonify({
        "success": False,
        "error": "Method not allowed",
        "timestamp": datetime.now().isoformat()
    }), 405

@app.errorhandler(RequestEntityTooLarge)
def file_too_large(error):
    return jsonify({
        "success": False,
        "error": "File too large. Maximum size: 100MB",
        "timestamp": datetime.now().isoformat()
    }), 413

@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        "success": False,
        "error": "Internal server error",
        "timestamp": datetime.now().isoformat()
    }), 500

# ============================================================================
# APPLICATION STARTUP & CLEANUP
# ============================================================================

def shutdown_handler():
    """Cleanup on shutdown"""
    cleanup_temp_files()
    logger.info("Flask app shutting down")

import atexit
atexit.register(shutdown_handler)

# ============================================================================
# MAIN APPLICATION ENTRY POINT (Production Ready)
# ============================================================================

# Initialize pipeline at module level for production
if not init_pipeline():
    logger.error("Failed to initialize pipeline on startup!")

if __name__ == '__main__':
    print("Legal Document AI Analyzer - Flask API Server")
    print("=" * 60)
    
    # Check if running in production
    is_production = os.environ.get('RENDER') or os.environ.get('FLASK_ENV') == 'production'
    
    if pipeline:
        print("Pipeline initialized successfully!")
        print(f"Temporary upload directory: {temp_upload_dir}")
        
        if not is_production:
            print("\nAPI Endpoints:")
            print("   GET  /health              - Health check")
            print("   GET  /status              - Pipeline status")
            print("   POST /upload              - Upload files")
            print("   POST /process             - Process documents")
            print("   POST /load_stores         - Load existing stores")
            print("   POST /query               - Query documents")
            print("   POST /query_file          - Query specific file")
            print("   POST /compare             - Compare categories")
            print("   POST /compare_files       - Compare specific files")
            print("   POST /compare_obligations - Compare obligations")
            print("   POST /compare_termination - Compare termination clauses")
            print("   POST /compare_clauses     - Compare specific clauses")
            print("   POST /summary             - Get document summary")
            print("   POST /summary_file        - Get file summary")
            print("   POST /obligations         - Find key obligations")
            print("   POST /obligations_file    - Find obligations in file")
            print("   POST /termination         - Find termination clauses")
            print("   POST /termination_file    - Find termination in file")
            print("   POST /explain_clause      - Explain specific clause")
            print("   POST /explain_clause_file - Explain clause in file")
            print("   GET  /categories          - Get available categories")
            print("   GET  /categorizations     - Get categorization results")
            print("   POST /export_report       - Export categorization report")
            print("   POST /conversation/clear  - Clear conversation history")
            print("   GET  /conversation/history - Get conversation history")
            print("   DELETE /stores/delete     - Delete category stores")
            
            print(f"\nStarting Flask development server...")
            print(f"API will be available at: http://localhost:5000")
        
        # Only run Flask dev server in development mode
        if not is_production:
            app.run(
                host='0.0.0.0',
                port=int(os.environ.get('PORT', 5000)),
                debug=False,  # Disable debug in production-like setup
                threaded=True
            )
        else:
            logger.info("Running in production mode - WSGI server will handle the app")
    else:
        print("Failed to initialize pipeline!")
        if not is_production:
            print("Check your configuration and try again.")
        exit(1)

# Make the app available for WSGI servers (like gunicorn)
application = app

@app.route('/log', methods=['GET'])
def show_logs():
    """Endpoint to display the contents of the log file."""
    try:
        log_file_path = 'logs/flask_api.log'
        if not os.path.exists(log_file_path):
            return jsonify({"success": False, "error": "Log file not found."}), 404

        with open(log_file_path, 'r') as log_file:
            logs = log_file.read()

        return jsonify({"success": True, "logs": logs.splitlines()}), 200
    except Exception as e:
        logger.error(f"Error reading log file: {e}")
        return jsonify({"success": False, "error": "Failed to read log file."}), 500