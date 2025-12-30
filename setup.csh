# Setup script for PredAI test environment
# Creates a virtual environment and installs dependencies
# Usage: ./setup.csh [python_version]
# Example: ./setup.csh python3.13

# Use provided Python version or default to python3.10
PYTHON_VERSION=${1:-python3.10}

echo "Using Python: $PYTHON_VERSION"
echo "Creating Python virtual environment..."
$PYTHON_VERSION -m venv venv

echo "Activating virtual environment..."
source venv/bin/activate

echo "Installing dependencies..."
python -m pip install --upgrade pip
python -m pip install -r predai/requirements.txt

echo ""
echo "Setup complete!"
echo "To activate the environment manually, run:"
echo "  source venv/bin/activate.csh"
echo ""
echo "To run tests, execute:"
echo "  ./run_all"

