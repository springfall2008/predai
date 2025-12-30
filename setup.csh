# Setup script for PredAI test environment
# Creates a virtual environment and installs dependencies

echo "Creating Python virtual environment..."
python3 -m venv venv

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

