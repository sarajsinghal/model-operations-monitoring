import argparse
import pandas as pd
from src.platform import predict_batch

parser = argparse.ArgumentParser(description="Score a feature CSV with the production model")
parser.add_argument("input", help="CSV containing feature_0 through feature_5")
parser.add_argument("--output", default="predictions.csv")
args = parser.parse_args()
result = predict_batch(pd.read_csv(args.input))
result.to_csv(args.output, index=False)
print(f"Wrote {len(result)} prediction rows to {args.output}")
