# CO₂ Emissions Dashboard

This repository contains the code for an interactive dashboard to explore
parcel-level CO₂ emissions under different water management scenarios.
The dashboard is built with Plotly Dash (Python).

## How to run the dashboard locally (Ubuntu terminal)

1. Open an Ubuntu terminal and navigate to the project folder.

2. Create a virtual environment:
```ruby
python -m venv venv\
```

3. Activate the virtual environment:
```ruby
source venv/bin/activate
```

4. Install the required dependencies:
```ruby
pip install -r requirements.txt
```

5. Unzip the data.zip file into a folder named `data`


6. Run the application:
```ruby
python app.py
```

7. Open the dashboard in your browser:
http://127.0.0.1:8050/

The dashboard will run locally on your machine.

## Notes
If additional system dependencies are required (e.g. for GeoPandas),
follow the instructions shown in the terminal error messages.