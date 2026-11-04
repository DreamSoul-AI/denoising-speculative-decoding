#!/bin/bash
pip install -q torch transformers accelerate numpy 2>/dev/null || pip3 install -q torch transformers accelerate numpy 2>/dev/null
python3 esp_v6.py || python esp_v6.py
