# Pokemon Legendary Classifier

A machine learning project that predicts whether a Pokemon is legendary based on its stats and type.

## Features

- Predicts legendary status using Pokemon's base stats and primary type
- Uses both Logistic Regression (baseline) and Random Forest with SMOTE
- Includes feature importance analysis
- Supports both individual Pokemon predictions and batch predictions via CSV
- Visualization of model performance metrics
- Command-line interface for easy predictions
- Comprehensive Jupyter notebooks for analysis and examples

## Project Structure
```
pokemon_ml/
├── data/
│   ├── pokemon.csv        # Main dataset
│   └── test_pokemon.csv   # Example Pokemon for testing
├── notebooks/
│   ├── 01_pokemon_analysis.ipynb    # Data analysis and visualization
│   ├── 02_pokemon_model.ipynb       # Model training and evaluation
│   ├── 03_sample_predictions.ipynb  # Example predictions and visualizations
│   ├── pokemon_model.py             # Main model implementation
│   └── test_predictions.py          # Prediction testing script
├── predict_pokemon.py      # Command-line interface
├── requirements.txt        # Project dependencies
└── README.md
```

## Features Used
- Base Stats:
  - HP
  - Attack
  - Defense
  - Special Attack
  - Special Defense
  - Speed
- Engineered Features:
  - Total Stats
  - Physical Average (Attack/Defense)
  - Special Average (Sp. Attack/Sp. Defense)
- Pokemon Type (one-hot encoded)

## Model Performance
- Random Forest with SMOTE for handling class imbalance
- Evaluation metrics include:
  - ROC-AUC Score
  - Precision-Recall
  - F1-Score
  - Confusion Matrix

## Usage

1. Clone the repository:
```bash
git clone [https://github.com/9lenn/pokemon_ml_analyzer/]
cd pokemon_ml_analyzer
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Use the Command-Line Interface:

   a. Single Pokemon prediction:
   ```bash
   python predict_pokemon.py single \
       --hp 100 \
       --attack 150 \
       --defense 140 \
       --sp-attack 120 \
       --sp-defense 100 \
       --speed 90 \
       --type1 dragon
   ```

   b. Batch prediction from CSV:
   ```bash
   python predict_pokemon.py batch --file data/test_pokemon.csv
   ```

4. Use in Python Code:
```python
from notebooks.pokemon_model import predict_legendary, rf_pipeline

pokemon = {
    'hp': 100,
    'attack': 150,
    'defense': 140,
    'sp_attack': 120,
    'sp_defense': 100,
    'speed': 90,
    'type1': 'dragon'
}

is_legendary, prob = predict_legendary(rf_pipeline, pokemon)
```

5. Explore the Notebooks:
- `01_pokemon_analysis.ipynb`: Data analysis and visualization
- `02_pokemon_model.ipynb`: Model training and evaluation
- `03_sample_predictions.ipynb`: Example predictions with visualizations

## Requirements
- Python 3.11+
- pandas
- numpy
- scikit-learn
- imbalanced-learn
- matplotlib
- seaborn
- jupyter

## Model Details

### Feature Importance
The Random Forest model identifies the most important features for legendary classification:
1. Total Stats
2. Special Attack
3. Attack
4. Speed
5. Defense

### Performance Metrics
- Accuracy: ~95%
- F1-Score: ~0.85
- ROC-AUC: ~0.92

## Future Improvements
- Add support for secondary types
- Include more feature engineering
- Experiment with other models (XGBoost, Neural Networks)
- Add web interface for easy predictions
- Include more visualization options

## Contributing
Feel free to open issues or submit pull requests with improvements! 
