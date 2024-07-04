#!/bin/bash

if [ -d "DTI-venv" ]; then
    echo "Virtual environment 'DTI-venv' activated."
else
    python3 -m venv DTI-venv
    pip install -r requirements.txt
    echo "Virtual environment 'DTI-venv' created."
    echo "Setup complete. Virtual environment 'DTI-venv' is ready with required packages installed."
fi

source DTI-venv/bin/activate



