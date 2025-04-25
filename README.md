# Football Analysis

This project uses computer vision to analyze football footage.

## Setup

1. Install dependencies
```bash
pip install -r requirements.txt
```

3. Add your Roboflow API key
   - Open `main.py`
   - Locate the line: `ROBOFLOW_API_KEY = 'your_key_here'`
   - Replace `'your_key_here'` with your actual Roboflow API key

4. Download model and set path 
Link: https://drive.google.com/file/d/1Nl-AoZASV9VZWJI02Pvzn20CuFFU5lgH/view?usp=drive_link
   - Locate the line: `PATHOFMODEL = 'set_model_path'`
   - Replace `'set_model_path'` with your actual Roboflow API key  

## Getting Your Roboflow API Key

1. Log in to your Roboflow account at [https://app.roboflow.com](https://app.roboflow.com)
2. Click on your profile in the top right corner
3. Select "Settings"
4. Your API key will be displayed or can be generated there

## Usage

Run the main script:
```bash
python main.py
```

## Features

- Player detection and tracking
- Ball tracking
- Event detection (passes, shots, etc.)
- Tactical analysis

## Requirements

- Python 3.7+
- Roboflow account
- Additional dependencies listed in requirements.txt
