# backend/start.sh
#!/bin/bash
pip install -r requirements.txt
hypercorn main:app --bind 0.0.0.0:8000