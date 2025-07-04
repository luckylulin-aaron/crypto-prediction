#!/usr/bin/env python3
"""
Startup script for the Crypto Trading Bot Web Server.
"""

import os
import sys

# Add the app directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

# Import and run the server
from app.server import app, initialize_client

if __name__ == '__main__':
    print("🚀 Starting Crypto Trading Bot Web Server...")
    
    # Initialize client
    if initialize_client():
        print("✅ Coinbase client initialized successfully")
        print("🌐 Server starting on http://localhost:8000")
        print("📊 Dashboard available at http://localhost:8000")
        print("🔧 API endpoints available at http://localhost:8000/api/")
        print("\nPress Ctrl+C to stop the server")
        
        # Run the Flask app
        app.run(host='0.0.0.0', port=8000, debug=False)
    else:
        print("❌ Failed to initialize Coinbase client")
        print("Please check your API credentials in app/screte.ini")
        sys.exit(1) 