"""
Script to run scoring on single summary
2024-08-20
"""

import json
import os
import time
from argparse import ArgumentParser, BooleanOptionalAction
from datetime import date

import pandas as pd
from dotenv import load_dotenv

from llm_political_analysis.modules.analyze import bulk_analyze_text

parser = ArgumentParser()
parser.add_argument("-m", "--model", dest="model", nargs="*", default=[],
                    help="Models to use. [gpt-3.5-turbo, gpt-4o, gpt-4, claude-3-5-sonnet-20240620, claude-3-opus-20240229, claude-3-sonnet-20240229, claude-3-haiku-20240307, gemini-1.5-pro-001]")
parser.add_argument("-o", "--output-dir", dest="output_dir",
                    help="The base output directory.")
parser.add_argument("-i", "--input-dir", nargs="*", dest="input_dir",
                    help="A list of the input directories. Can be both full text or summary. Text files in the directories will be pass to the analyze function.")
parser.add_argument("-t", "--tag", dest="tag", default="",
                    help="Tag for this run.")
parser.add_argument("-n", "--no-subfolder",
                    action=BooleanOptionalAction, dest="no_subfolder", default=False,
                    help="Don't create a subfolder in the output directory.")
parser.add_argument("-v", "--prompt-version", dest="prompt_version", default="",
                    help="Prompt version")
parser.add_argument("-p", "--override-persona-and-encouragement", type=int,
                    dest="override_persona_and_encouragement", default=[0, 1], nargs="*",
                    help="Override persona and encouragement. Should be two integers of the index of persona and encouragement.")
parser.add_argument("-g", "--debug",
                    action=BooleanOptionalAction, dest="debug", default=False,
                    help="Debug flag.")

model_name_alias = {
    "gpt": "gpt-4o",
    "claude": "claude-3-5-sonnet-20240620",
    "gemini": "gemini-1.5-pro-001"
}


def parse_issue_from_filepath(filepath_: str):
    prefix = os.path.basename(filepath_).split("__")[0]
    for issue_name in ['european_union', 'taxation', 'lifestyle', 'immigration', 'environment', 'decentralization']:
        if prefix.endswith(issue_name):
            return issue_name


if __name__ == "__main__":
    # Load API keys from env
    load_dotenv()

    # Timestamp as the output key
    timestamp = f"{date.today()}-{int(time.time())}"
    # Get all run arguments
    args = parser.parse_args()
    print(args)
    output_dir = os.path.join(
        os.path.abspath(os.path.dirname(os.path.dirname(__file__))), "data", "output"
    ) if args.output_dir is None else args.output_dir
    if not args.no_subfolder:
        output_name = f"{timestamp}{'-'+args.tag if args.tag else ''}"
        output_dir = os.path.join(output_dir, output_name)
        os.makedirs(output_dir)
    models = [model_name_alias.get(name, name) for name in args.model] if args.model else args.model
    prompt_version = args.prompt_version
    debug = args.debug
    override_persona_and_encouragement = args.override_persona_and_encouragement

    # Create the output folder and log run args
    args_dict = args.__dict__
    args_dict["model"] = models
    args_dict["timestamp"] = timestamp
    args_dict["output_dir"] = output_dir

    with open(os.path.join(output_dir, "args.json"), "a", encoding="utf-8") as f:
        f.write(json.dumps(args_dict))
        f.write("\n")

    # Get absolute path of input files
    input_file_list = []
    for input_dir in args.input_dir:
        for filename in os.listdir(input_dir):
            if filename.endswith(".txt"):
                input_file_list.append(
                    os.path.join(input_dir, filename)
                )

    existing_results_filepath = os.path.join(output_dir, "analyze_results.xlsx")
    existing_output_df = None
    if os.path.exists(existing_results_filepath):
        existing_output_df = pd.read_excel(existing_results_filepath)

    for filepath in input_file_list:
        models_to_analyze = models
        issue_to_analyze = parse_issue_from_filepath(filepath)
        if existing_output_df is not None:
            res = existing_output_df[(existing_output_df["file"]==filepath)&(existing_output_df["issue"]==issue_to_analyze)]
            models_to_analyze = list(set(models_to_analyze).difference(res["model"].tolist()))
        if len(models_to_analyze) > 0:
            # for model in models_to_analyze:
            #     bulk_analyze_text(
            #         [filepath],
            #         models_to_analyze,
            #         [issue_to_analyze],
            #         summarize=False,
            #         override_persona_and_encouragement=override_persona_and_encouragement,
            #         parse_retries=0,
            #         output_dir=output_dir
            #     )
            bulk_analyze_text(
                [filepath],
                models_to_analyze,
                [issue_to_analyze],
                summarize=False,
                override_persona_and_encouragement=override_persona_and_encouragement,
                parse_retries=0,
                output_dir=output_dir
            )
        else:
            print(f"Skipping: {[issue_to_analyze, os.path.basename(filepath)]}")
