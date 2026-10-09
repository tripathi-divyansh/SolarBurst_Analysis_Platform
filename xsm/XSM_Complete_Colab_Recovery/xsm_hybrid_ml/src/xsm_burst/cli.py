import argparse
import json
from pathlib import Path
import pandas as pd
from .io import load_lightcurve
from .schema import PipelineConfig
from .pipeline import analyze
from .models import train_models, BurstModels
from .serialization import save_analysis, write_json
from .synthetic import run_demo


def main(argv=None):
    p = argparse.ArgumentParser(description="XSM hybrid detection and XGBoost models")
    sub = p.add_subparsers(dest="command", required=True)
    extract = sub.add_parser("extract", help="Parse observation, detect candidates, fit and extract features")
    extract.add_argument("observation")
    extract.add_argument("--mapping", required=True)
    extract.add_argument("--group", required=True, help="Independent observing-period group, not a candidate ID")
    extract.add_argument("--config")
    extract.add_argument("--out", required=True)
    train = sub.add_parser("train", help="Train using explicit disjoint observing-period partitions")
    train.add_argument("candidates")
    train.add_argument("--exposure", required=True)
    train.add_argument("--model", required=True)
    train.add_argument("--target-far", type=float, default=1.0)
    train.add_argument("--trees", type=int, default=400)
    predict = sub.add_parser("predict", help="Score an extracted feature table")
    predict.add_argument("candidates")
    predict.add_argument("--model", required=True)
    predict.add_argument("--trust-model", action="store_true")
    predict.add_argument("--out", required=True)
    demo = sub.add_parser("demo", help="Run a clearly labeled synthetic end-to-end demonstration")
    demo.add_argument("--out", default="demo_output")
    demo.add_argument("--groups", type=int, default=40)
    args = p.parse_args(argv)
    if args.command == "extract":
        cfg = PipelineConfig(**json.loads(Path(args.config).read_text())) if args.config else PipelineConfig()
        result = analyze(load_lightcurve(args.observation, args.mapping), cfg, args.group)
        save_analysis(result, args.out)
        print(f"Saved {len(result['catalog'])} unscored candidates to {args.out}")
    elif args.command == "train":
        bundle, report = train_models(pd.read_csv(args.candidates), pd.read_csv(args.exposure),
                                      target_far=args.target_far, n_estimators=args.trees)
        bundle.save(args.model)
        write_json(Path(args.model).with_suffix(".evaluation.json"), report)
        print(json.dumps(report, indent=2))
    elif args.command == "predict":
        model = BurstModels.load(args.model, trusted=args.trust_model)
        result = model.score(pd.read_csv(args.candidates))
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(args.out, index=False)
        print(f"Saved {len(result)} candidate decisions to {args.out}")
    else:
        report = run_demo(args.out, args.groups)
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
