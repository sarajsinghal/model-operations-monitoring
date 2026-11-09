import argparse
import json
import pandas as pd
from src.platform import (attach_labels, audit_log, export_report, models,
                          set_status, simulate_events)

p = argparse.ArgumentParser(description="Manage the local model operations demo")
sub = p.add_subparsers(dest="command", required=True)
sub.add_parser("list")
status = sub.add_parser("status"); status.add_argument("version"); status.add_argument("value", choices=["candidate", "validated", "production", "archived"])
sim = sub.add_parser("simulate"); sim.add_argument("--rows", type=int, default=120); sim.add_argument("--no-drift", action="store_true")
labels = sub.add_parser("labels"); labels.add_argument("csv")
sub.add_parser("audit")
sub.add_parser("report")
a = p.parse_args()
if a.command == "list":
    print(models().to_string(index=False))
elif a.command == "status":
    set_status(a.version, a.value); print(f"{a.version}: {a.value}")
elif a.command == "simulate":
    print(simulate_events(a.rows, drift=not a.no_drift).head().to_string(index=False))
elif a.command == "labels":
    print(f"Attached {attach_labels(pd.read_csv(a.csv))} labels")
elif a.command == "audit":
    print(audit_log().to_string(index=False))
elif a.command == "report":
    print(export_report().decode())
