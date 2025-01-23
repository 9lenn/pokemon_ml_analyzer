#!/usr/bin/env python3
import argparse
import pandas as pd
from notebooks.pokemon_model import predict_legendary, predict_multiple_pokemon, rf_pipeline

def predict_from_args(args):
    """Make prediction from command line arguments."""
    pokemon = {
        'hp': args.hp,
        'attack': args.attack,
        'defense': args.defense,
        'sp_attack': args.sp_attack,
        'sp_defense': args.sp_defense,
        'speed': args.speed,
        'type1': args.type1.lower()
    }
    
    is_legendary, prob = predict_legendary(rf_pipeline, pokemon)
    print("\nPrediction Results:")
    print("-" * 60)
    print(f"Stats: {pokemon}")
    print(f"Prediction: {'Legendary' if is_legendary else 'Not Legendary'}")
    print(f"Probability of being legendary: {prob:.2%}")

def predict_from_csv(file_path):
    """Make predictions from a CSV file."""
    try:
        data = pd.read_csv(file_path)
        pokemon_list = data.to_dict('records')
        results = predict_multiple_pokemon(rf_pipeline, pokemon_list)
        return results
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description='Predict if a Pokemon is legendary based on its stats.')
    
    # Create subparsers for different modes
    subparsers = parser.add_subparsers(dest='mode', help='Prediction mode')
    
    # Single Pokemon prediction
    single_parser = subparsers.add_parser('single', help='Predict for a single Pokemon')
    single_parser.add_argument('--hp', type=int, required=True, help='HP stat')
    single_parser.add_argument('--attack', type=int, required=True, help='Attack stat')
    single_parser.add_argument('--defense', type=int, required=True, help='Defense stat')
    single_parser.add_argument('--sp-attack', type=int, required=True, help='Special Attack stat')
    single_parser.add_argument('--sp-defense', type=int, required=True, help='Special Defense stat')
    single_parser.add_argument('--speed', type=int, required=True, help='Speed stat')
    single_parser.add_argument('--type1', type=str, required=True, help='Primary type')
    
    # Batch prediction from CSV
    batch_parser = subparsers.add_parser('batch', help='Predict for multiple Pokemon from CSV')
    batch_parser.add_argument('--file', type=str, required=True, help='Path to CSV file')
    
    args = parser.parse_args()
    
    if args.mode == 'single':
        predict_from_args(args)
    elif args.mode == 'batch':
        results = predict_from_csv(args.file)
        if results:
            print("\nBatch Prediction Results:")
            print("-" * 60)
            for result in results:
                name = result['stats'].get('name', 'Unknown')
                pred = result['prediction']
                prob = result['probability']
                print(f"\nPokemon: {name}")
                print(f"Prediction: {pred}")
                print(f"Probability: {prob:.2%}")
    else:
        parser.print_help()

if __name__ == '__main__':
    main() 