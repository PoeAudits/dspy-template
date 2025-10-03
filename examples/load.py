import dspy
from dspy.datasets import DataLoader
import os
from pathlib import Path
from typing import Mapping

def load_data(
    data,
    fields: list[str] | tuple[str] | None = None,
    input_keys: tuple[str] = (),
    **kwargs,
) -> list[dspy.Example] | Mapping[str, list[dspy.Example]]:
    """
    Load data using the appropriate DataLoader method based on input type.

    Args:
        data: Can be a pandas DataFrame, file path (str/Path), or HuggingFace dataset name
        fields: Fields to extract from the dataset
        input_keys: Keys to mark as inputs in dspy.Example
        **kwargs: Additional arguments passed to the underlying loader method

    Returns:
        List of dspy.Example objects or a mapping of split names to lists of examples
    """
    loader = DataLoader()

    # Handle pandas DataFrame
    try:
        import pandas as pd
        if isinstance(data, pd.DataFrame):
            assert not isinstance(fields, tuple) 
            return loader.from_pandas(data, fields=fields, input_keys=input_keys)
    except ImportError:
        pass

    # Handle file paths or dataset names
    if isinstance(data, (str, Path)):
        data_str = str(data)

        # Check if it's a file path
        if os.path.exists(data_str):
            file_ext = Path(data_str).suffix.lower()

            assert not isinstance(fields, tuple) 
            if file_ext == ".csv":
                return loader.from_csv(data_str, fields=fields, input_keys=input_keys)
            elif file_ext == ".json":
                return loader.from_json(data_str, fields=fields, input_keys=input_keys)
            elif file_ext == ".parquet":
                return loader.from_parquet(data_str, fields=fields, input_keys=input_keys)
            else:
                raise ValueError(
                    f"Unsupported file type: {file_ext}. "
                    "Supported types: .csv, .json, .parquet"
                )

        # Assume it's a HuggingFace dataset name
        # Convert fields to tuple if provided
        fields_tuple = tuple(fields) if fields else None
        return loader.from_huggingface(
            data_str,
            fields=fields_tuple,
            input_keys=input_keys,
            **kwargs
        )

    # Handle if data is already a list of dspy.Example
    if isinstance(data, list) and all(isinstance(item, dspy.Example) for item in data):
        return data

    raise TypeError(
        f"Unsupported data type: {type(data)}. "
        "Expected pandas DataFrame, file path (str/Path), HuggingFace dataset name, "
        "or list of dspy.Example objects."
    )


def combine_batch_results(examples: list[dspy.Example], results: list[dspy.Prediction]):
    examples_dict = [example.toDict() for example in examples]
    results_dict = [result.toDict() for result in results]
    combined = [
        {**ex, **result}
        for ex, result in zip(examples_dict, results_dict)
    ]
    return combined



