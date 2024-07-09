#!/bin/bash

if [ -d "DTI-venv" ]; then
    echo "Virtual environment 'DTI-venv' activated."
    source DTI-venv/bin/activate
else
    python3 -m venv DTI-venv
    source DTI-venv/bin/activate
    pip install -r requirements.txt
    echo "Virtual environment 'DTI-venv' created."
    echo "Setup complete. Virtual environment 'DTI-venv' is ready with required packages installed."
fi





