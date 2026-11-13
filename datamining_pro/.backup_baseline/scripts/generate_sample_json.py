"""
Generates high-fidelity sample website activity JSON file for automated workflow testing.
"""
import json
import os
import pandas as pd

def create_sample_json():
    csv_path = "data/clickstream_sample.csv"
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        # Take first 400 events across ~90 sessions
        df_sub = df.head(450).copy()
        records = df_sub.to_dict(orient='records')
    else:
        records = [
            {
                "session_id": "SESS_0001",
                "timestamp": "2024-03-01 08:15:00",
                "page_name": "home",
                "page_path": "/home",
                "page_category": "Home",
                "event_type": "page_view",
                "dwell_time_seconds": 35
            },
            {
                "session_id": "SESS_0001",
                "timestamp": "2024-03-01 08:15:40",
                "page_name": "catalog_electronics",
                "page_path": "/catalog_electronics",
                "page_category": "Catalog",
                "event_type": "page_view",
                "dwell_time_seconds": 75
            },
            {
                "session_id": "SESS_0001",
                "timestamp": "2024-03-01 08:17:00",
                "page_name": "cart",
                "page_path": "/cart",
                "page_category": "Cart",
                "event_type": "add_to_cart",
                "dwell_time_seconds": 45
            },
            {
                "session_id": "SESS_0001",
                "timestamp": "2024-03-01 08:17:50",
                "page_name": "order_confirmation",
                "page_path": "/order_confirmation",
                "page_category": "Checkout",
                "event_type": "purchase",
                "dwell_time_seconds": 25
            }
        ]
        
    out_path = "data/sample_website_activity.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)
    print(f"Created {out_path} with {len(records)} events.")

if __name__ == '__main__':
    create_sample_json()
