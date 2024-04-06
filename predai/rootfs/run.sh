echo "Running App Process Management"
echo "Your API key is: $SUPERVISOR_TOKEN"

printenv

python3 /startup.py $SUPERVISOR_TOKEN
