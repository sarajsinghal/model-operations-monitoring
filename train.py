import argparse
from src.platform import train_model

parser = argparse.ArgumentParser(description="Train a versioned local classifier")
parser.add_argument("--seed", type=int, default=41)
args = parser.parse_args()
result = train_model(args.seed)
print(f"Version: {result['version']}\nMetrics: {result['metrics']}\nArtifact: {result['artifact_path']}")
print("Status: candidate (validate and promote explicitly)")
