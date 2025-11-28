#!/usr/bin/env python3
"""
View prompt from a saved experiment
Usage: python3 view_prompt.py <path_to_pkl_file>
"""
import sys
import pickle


def view_prompt(pkl_file):
    """Load and display prompt from pickle file"""
    try:
        with open(pkl_file, 'rb') as f:
            data = pickle.load(f)

        print("="*70)
        print(f"Experiment: {pkl_file}")
        print("="*70)
        print()

        # Display metadata
        print("Metadata:")
        print(f"  Model: {data.get('model', 'N/A')}")
        print(f"  Prompt Type: {data.get('prompt_type', 'N/A')}")
        print(f"  Instruction: {data.get('instruction', 'N/A')}")
        print(f"  Waypoints: {len(data.get('episode_0', []))}")
        print()

        # Display prompt
        if 'prompt' in data:
            print("="*70)
            print("PROMPT:")
            print("="*70)
            print(data['prompt'])
            print("="*70)
        else:
            print("⚠️  No prompt saved in this file")
            print()

        # Display raw response (first 500 chars)
        if 'raw_response' in data:
            print()
            print("="*70)
            print("RAW RESPONSE (first 500 chars):")
            print("="*70)
            response = data['raw_response']
            print(response[:500])
            if len(response) > 500:
                print("...")
            print("="*70)

    except FileNotFoundError:
        print(f"❌ File not found: {pkl_file}")
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 view_prompt.py <path_to_pkl_file>")
        print()
        print("Available files:")
        print("  - results/qwen3_improved_prompt_path.pkl")
        print("  - results/qwen3_trajectory_prompt_path.pkl")
        sys.exit(1)

    view_prompt(sys.argv[1])
