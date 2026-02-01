import subprocess
import sys
import os
from pathlib import Path

def load_env_file():
    """Load environment variables from .env file if it exists"""
    env_file = Path(".env")
    if env_file.exists():
        print("📄 Loading environment variables from .env file...")
        with open(env_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    try:
                        key, value = line.split('=', 1)
                        os.environ[key.strip()] = value.strip()
                    except ValueError:
                        continue
        print("✅ Environment variables loaded")
    else:
        print("⚠️  No .env file found, using default values")

def main():
    if not os.path.exists('app.py'):
        print("❌ Error: app.py not found. Please run this script from the project root directory.")
        sys.exit(1)
    
    # Load environment variables
    load_env_file()
    
    # Display configuration
    fhir_base_url = os.getenv("FHIR_BASE_URL", "http://65.0.127.208:30007/fhir")
    default_group_id = os.getenv("DEFAULT_GROUP_ID", "Lepto")
    
    print("=" * 60)
    print("🚀 Starting Healthcare Analytics Dashboard")
    print("=" * 60)
    print(f"📊 FHIR Base URL: {fhir_base_url}")
    print(f"🏥 Default Group ID: {default_group_id}")
    print(f"🌐 Opening browser at http://localhost:8050")
    print(f"🛑 Press Ctrl+C to stop the application")
    print("=" * 60)
    
    try:
        subprocess.run([
            sys.executable, '-m', 'streamlit', 'run', 'app.py',
            '--server.port', '8050',
            '--server.address', '0.0.0.0',
            '--browser.gatherUsageStats', 'false',
            '--theme.base', 'light'
        ])
    except KeyboardInterrupt:
        print("\n🛑 Application stopped by user")
    except Exception as e:
        print(f"❌ Error running application: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()