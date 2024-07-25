import json
import os
import time
from argparse import ArgumentParser, BooleanOptionalAction
from datetime import date

from dotenv import load_dotenv

from llm_political_analysis.modules.analyze import bulk_analyze_text
from llm_political_analysis.modules.summarize import summarize_file

parser = ArgumentParser()
parser.add_argument("-m", "--model", dest="model", nargs="*", default=[],
                    help="Models to use. [gpt-3.5-turbo, gpt-4o, gpt-4, claude-3-5-sonnet-20240620, claude-3-opus-20240229, claude-3-sonnet-20240229, claude-3-haiku-20240307, gemini-1.5-pro-001]")
parser.add_argument("-d", "--dimension (issue areas)", dest="issue_areas", nargs="*", default=['european_union', 'taxation', 'lifestyle', 'immigration', 'environment', 'decentralization'],
                    help="Issue areas to analyze. ['european_union', 'taxation', 'lifestyle', 'immigration', 'environment', 'decentralization']")
parser.add_argument("-s", "--summary",
                    action=BooleanOptionalAction, dest="summarize", default=False,
                    help="Whether to summarize text.")
parser.add_argument("-a", "--analyze",
                    action=BooleanOptionalAction, dest="analyze", default=False,
                    help="Whether to analyze text.")
parser.add_argument("-o", "--output-dir", dest="output_dir",
                    help="The output directory.")
parser.add_argument("-i", "--input-dir", nargs="*", dest="input_dir",
                    help="A list of the input directories. Can be both full text or summary. Text files in the directories will be pass to the analyze function.")
parser.add_argument("-t", "--tag", dest="tag", default="",
                    help="Tag for this run.")
parser.add_argument("-p", "--override-persona-and-encouragement", type=int,
                    dest="override_persona_and_encouragement", default=None, nargs="*",
                    help="Override persona and encouragement. Should be two integers of the index of persona and encouragement.")
parser.add_argument("-r", "--parse_retries", type=int,
                    dest="parse_retries", default=3,
                    help="The number of times to retry parsing the response.")


model_name_alias = {
    "gpt": "gpt-4o",
    "claude": "claude-3-5-sonnet-20240620",
    "gemini": "gemini-1.5-pro-001"
}


if __name__ == "__main__":
    # Load API keys from env
    load_dotenv()

    # Get all run arguments
    args = parser.parse_args()

    # Timestamp as the output key
    timestamp = f"{date.today()}-{int(time.time())}"
    output_name = f"{timestamp}{'-'+args.tag if args.tag else ''}"
    output_dir = os.path.join(
        os.path.abspath(os.path.dirname(os.path.dirname(__file__))), "data", "output", output_name
    ) if args.output_dir is None else os.path.join(args.output_dir, output_name)
    models = [model_name_alias.get(name, name) for name in args.model] if args.model else args.model

    # Create the output folder and log run args
    args_dict = args.__dict__
    args_dict["model"] = models
    args_dict["timestamp"] = timestamp
    args_dict["output_dir"] = output_dir
    os.makedirs(output_dir)
    with open(os.path.join(output_dir, "args.json"), "w", encoding="utf-8") as f:
        f.write(json.dumps(args_dict))

    # Get absolute path of input files
    input_file_list = []
    for input_dir in args.input_dir:
        for filename in os.listdir(input_dir):
            if filename.endswith(".txt"):
                input_file_list.append(
                    os.path.join(input_dir, filename)
                )

    if args.analyze:
        # Analyze, do summarize base on args.summarize
        bulk_analyze_text(
            input_file_list,
            models,
            args.issue_areas,
            output_dir,
            summarize=args.summarize,
            override_persona_and_encouragement=args.override_persona_and_encouragement,
            parse_retries=args.parse_retries
        )
    elif not args.analyze and args.summarize:
        # Summarize only
        for filepath in input_file_list:
            summarize_file(
                filepath, args.issue_areas, output_dir
            )
