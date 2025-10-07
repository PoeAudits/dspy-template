from lms import *
from examples import load_data, combine_batch_results
import logging
from log import setup_logging

setup_logging(level=logging.INFO)

def main():
    # examples = load_data("examples/data/example.csv", fields=["text", "label"], input_keys=("message",)) 
    pass


if __name__ == "__main__":
    main()
