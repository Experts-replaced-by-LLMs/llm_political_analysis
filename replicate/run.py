"""
Script to replicate all runs
2024-10-02
"""

import json
import os
import time
from argparse import ArgumentParser, BooleanOptionalAction
from datetime import date

from dotenv import load_dotenv

from llm_political_analysis.modules.analyze import analyze_summary
from llm_political_analysis.modules.summarize import summarize_dataset
from llm_political_analysis.modules.utils import list_files

parser = ArgumentParser()
parser.add_argument("-c", "--config", dest="config", nargs="*", default=[],
                    help="Which config to run. Config name should be config id like xxx_01.")
parser.add_argument("-p", "--prototyping", action=BooleanOptionalAction,
                    dest="prototyping", default=False,
                    help="Run all prototyping. Override --config.")
parser.add_argument("-o", "--output-dir", dest="output_dir",
                    help="The output directory.")
parser.add_argument("-f", "--output-filenmame", dest="output_filename", default="summaries.csv",
                    help="The name of the output file.")
parser.add_argument("-d", "--dataset_path", dest="dataset_path",
                    help="The manifesto dataset file path.")
parser.add_argument("-g", "--debug",
                    action=BooleanOptionalAction, dest="debug", default=False,
                    help="Debug flag.")
parser.add_argument("-y", "--dry_run",
                    action=BooleanOptionalAction, dest="dry_run", default=False,
                    help="Dry run flag.")

def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.loads(f.read())
        # summary configs
        summary_args = {
            "summary_size": cfg.get("summary_size"),
            "max_tokens_factor": cfg.get("max_tokens_factor"),
            "prompt_version": cfg.get("prompt_version")
        }
        summary_args = {k:v for k, v in summary_args.items() if v is not None}
        summary_model_args = cfg.get("summary_model_args", {})

        all_summary_model_args = {}

        for model in cfg["summary_model"]:
            model_args = summary_model_args.get(model, {})
            all_summary_model_args[model] = {**summary_args, **model_args}

        analyze_model_args = {
            "model": cfg["model"],
            "use_few_shot": cfg.get("use_few_shot", False),
            "override_persona_and_encouragement": cfg.get("override_persona_and_encouragement", None)
        }

        all_config = {
            "summary_model_args": all_summary_model_args,
            "analyze_model_args": analyze_model_args,
            "tag": cfg["tag"],
            "dataset": cfg["dataset"],
            "all_issue": cfg.get("all_issue", False),
            "use_summary_from": cfg.get("use_summary_from", None),
            "issue_areas": cfg["issue_areas"],
        }

        return all_config

if __name__ == "__main__":
    # Load API keys from env
    load_dotenv()

    # Log timestamp
    timestamp = f"{date.today()}-{int(time.time())}"
    # Get replicate arguments
    args = parser.parse_args()
    output_dir = args.output_dir
    output_filename = args.output_filename
    dataset_path = args.dataset_path
    prototyping = args.prototyping
    config_to_use = args.config
    if prototyping:
        config_to_use = [
            "prototyping_01", "prototyping_02", "prototyping_03", "prototyping_04",
            "prototyping_05", "prototyping_06", "prototyping_07"
        ]
    debug = args.debug
    dry_run = args.dry_run

    # Set default dataset filepath
    if not dataset_path:
        dataset_path = os.path.join(os.path.abspath(os.path.dirname(os.path.dirname(__file__))), "data", "manifestos.csv")

    # Build output folder
    output_dir = os.path.join(
        os.path.abspath(os.path.dirname(os.path.dirname(__file__))), "data", "output", "all_results"
    ) if output_dir is None else output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_filename = args.output_filename

    # Load configs
    config_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), "configs")
    config_filenames = list_files(config_dir, extension=".json")
    config_filenames = [
        os.path.join(config_dir, filename)
        for filename in config_filenames
        if os.path.splitext(filename)[0].split("__")[0] in config_to_use
    ]
    config_filenames.sort()
    use_configs = [
        load_config(config_filename)
        for config_filename in config_filenames
    ]

    for use_cfg in use_configs:
        issue_areas = use_cfg["issue_areas"]
        summary_model_args = use_cfg["summary_model_args"]
        use_tag = use_cfg.get("use_summary_from") or use_cfg["tag"]
        for model_name, model_args in summary_model_args.items():
            if use_cfg["all_issue"]:
                summarize_dataset(
                    dataset_path, use_cfg["dataset"], issue_areas, output_dir=output_dir, output_filename=output_filename,
                    model=model_name, tag=use_tag, debug=debug, dry_run=dry_run, save_log=True, **model_args
                )
            else:
                for issue in issue_areas:
                    summarize_dataset(
                        dataset_path, use_cfg["dataset"], issue, output_dir=output_dir, output_filename=output_filename,
                        model=model_name, tag=use_tag, debug=debug, dry_run=dry_run, save_log=True, **model_args
                    )
        # Analyzing summaries
        analyze_model_args = use_cfg["analyze_model_args"]
        model = analyze_model_args["model"]
        override_persona_and_encouragement = analyze_model_args["override_persona_and_encouragement"]
        use_few_shot=analyze_model_args["use_few_shot"]
        analyze_summary(
            os.path.join(output_dir, output_filename), model, tag=use_tag,
            override_persona_and_encouragement=override_persona_and_encouragement, use_few_shot=use_few_shot,
            output_dir=output_dir, dry_run=dry_run
        )
