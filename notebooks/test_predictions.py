import pandas as pd
import os
from pokemon_model import predict_legendary, predict_multiple_pokemon, rf_pipeline

# Get the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))
test_data_path = os.path.join(os.path.dirname(current_dir), 'data', 'test_pokemon.csv')

# 1. Test individual Pokemon
print("Testing Individual Pokemon:")
print("-" * 60)

# Mewtwo (Legendary Psychic-type)
mewtwo = {
    'hp': 106,
    'attack': 110,
    'defense': 90,
    'sp_attack': 154,
    'sp_defense': 90,
    'speed': 130,
    'type1': 'psychic'
}

is_legendary, prob = predict_legendary(rf_pipeline, mewtwo)
print("\nMewtwo Test:")
print(f"Prediction: {'Legendary' if is_legendary else 'Not Legendary'}")
print(f"Probability of being legendary: {prob:.2%}")

# 2. Test using CSV file
print("\nTesting Pokemon from CSV:")
print("-" * 60)

# Read the CSV file
test_data = pd.read_csv(test_data_path)

# Convert to list of dictionaries
pokemon_list = test_data.to_dict('records')

# Make predictions
results = predict_multiple_pokemon(rf_pipeline, pokemon_list)

# Print summary
print("\nSummary:")
print("-" * 60)
for result in results:
    name = result['stats']['name']
    pred = result['prediction']
    prob = result['probability']
    print(f"{name}: {pred} (Probability: {prob:.2%})") 