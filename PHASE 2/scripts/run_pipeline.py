import argparse, json
from pathlib import Path
from preprocessing.news_loader import load_news
from feature_engineering.build_features import enrich_events
from models.predictor import DualPredictor
from alerts.alert_generator import build_alert, render_text
from config import DATA_PROCESSED

def main(input_path):
    news = load_news(input_path)
    enriched = enrich_events(news)
    predictor = DualPredictor()
    alerts = []
    for row in enriched.to_dict(orient="records"):
        preds = predictor.predict(row)
        alert = build_alert(row,preds)
        alerts.append(alert)
    DATA_PROCESSED.mkdir(parents=True,exist_ok=True)
    csv_path = DATA_PROCESSED/"event_analysis.csv"
    jsonl_path = DATA_PROCESSED/"investor_alerts.jsonl"
    enriched.to_csv(csv_path,index=False)
    with jsonl_path.open("w",encoding="utf-8") as f:
        for a in alerts: f.write(json.dumps(a,default=str)+"\n")
    print(f"Analysed {len(enriched)} news row(s).")
    print("Event analysis:",csv_path)
    print("Investor alerts:",jsonl_path)
    for a in alerts[:10]:
        print("\n"+render_text(a))

if __name__ == "__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--input",required=True)
    a=p.parse_args()
    main(a.input)
