"""
Script to run summary only
2024-08-20
"""

import json
import os
import time
from argparse import ArgumentParser, BooleanOptionalAction
from datetime import date
from dotenv import load_dotenv

from llm_political_analysis.modules.summarize import summarize_file

parser = ArgumentParser()
parser.add_argument("-m", "--model", dest="model", nargs="*", default=[],
                    help="Models to use. [gpt-3.5-turbo, gpt-4o, gpt-4, claude-3-5-sonnet-20240620, claude-3-opus-20240229, claude-3-sonnet-20240229, claude-3-haiku-20240307, gemini-1.5-pro-001]")
parser.add_argument("-d", "--dimension (issue areas)", dest="issue_areas", nargs="*", default=['european_union', 'taxation', 'lifestyle', 'immigration', 'environment', 'decentralization'],
                    help="Issue areas to analyze. ['european_union', 'taxation', 'lifestyle', 'immigration', 'environment', 'decentralization']")
parser.add_argument("-a", "--all",
                    action=BooleanOptionalAction, dest="all_issue", default=False,
                    help="Whether to summarize text on all issues all at once.")
parser.add_argument("-o", "--output-dir", dest="output_dir",
                    help="The base output directory.")
parser.add_argument("-i", "--input-dir", nargs="*", dest="input_dir",
                    help="A list of the input directories. Can be both full text or summary. Text files in the directories will be pass to the analyze function.")
parser.add_argument("-t", "--tag", dest="tag", default="",
                    help="Tag for this run.")
parser.add_argument("-n", "--no-subfolder",
                    action=BooleanOptionalAction, dest="no_subfolder", default=False,
                    help="Don't create a subfolder in the output directory.")
parser.add_argument("-s", "--size", type=int,
                    dest="summary_size", default=[300, 400], nargs="*",
                    help="Summary size. Default to [300, 400]")
parser.add_argument("-f", "--factor", type=float,
                    dest="max_tokens_factor", default=1.0,
                    help="Max tokens factor. Default to 1.0")
parser.add_argument("-c", "--chunk-size", type=int,
                    dest="chunk_size", default=100000,
                    help="Summary chunk size. Set 0 to disable chunking.")
parser.add_argument("-v", "--prompt-version", dest="prompt_version", default="",
                    help="Prompt version")
parser.add_argument("-g", "--debug",
                    action=BooleanOptionalAction, dest="debug", default=False,
                    help="Debug flag.")
parser.add_argument("-l", "--log",
                    action=BooleanOptionalAction, dest="save_log", default=True,
                    help="Save log.")

model_name_alias = {
    "gpt": "gpt-4o",
    "claude": "claude-3-5-sonnet-20240620",
    "gemini": "gemini-1.5-pro-002"
}


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
    summary_size = args.summary_size
    max_tokens_factor = args.max_tokens_factor
    chunk_size = args.chunk_size
    all_issue = args.all_issue
    issue_areas = args.issue_areas
    prompt_version = args.prompt_version
    debug = args.debug
    save_log = args.save_log

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

    for filepath in input_file_list:
        for model in models:
            try:
                if all_issue:
                    summarize_file(
                        filepath, issue_areas, output_dir, summary_size=summary_size, model=model,
                        chunk_size=chunk_size,
                        max_tokens_factor=max_tokens_factor, prompt_version=prompt_version, debug=debug,
                        if_exists="reuse", save_log=save_log
                    )
                else:
                    for issue in issue_areas:
                        summarize_file(
                            filepath, issue, output_dir, summary_size=summary_size, model=model,
                            chunk_size=chunk_size,
                            max_tokens_factor=max_tokens_factor, prompt_version=prompt_version, debug=debug,
                            if_exists="reuse", save_log=save_log
                        )
            except Exception as e:
                with open(os.path.join(output_dir, "args.json"), "a", encoding="utf-8") as f:
                    f.write(json.dumps({"model": model, "file": os.path.basename(filepath), "error": str(e)}))
                    f.write("\n")
        
