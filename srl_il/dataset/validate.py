import yaml
import os
import numpy as np

def validate_yaml_structure(file_path, expected_keys):
    """
    Validate the structure of a YAML file.
    Ensure that it contains the expected keys and data types.
    
    Args:
        file_path (str): Path to the YAML file to validate.
        expected_keys (dict): A dictionary with keys as YAML top-level keys
                              and expected value types (e.g., 'obs', 'pressure', 'pose').
    Returns:
        bool: True if the YAML structure is valid, False otherwise.
    """
    try:
        # Open the YAML file and load its content
        with open(file_path, 'r') as file:
            content = file.read()

        print(f"Validating YAML file: {file_path}")
        print(f"Content: {content[:100]}...")  # Display the first 100 characters for inspection

        # Load YAML data with a custom loader to handle 'array.array' type
        def array_constructor(loader, node):
            return np.array(node.value)

        yaml.add_constructor('tag:yaml.org,2002:python/object/apply:array.array', array_constructor)

        data = yaml.safe_load(content)

        # Check the structure of the YAML data
        if not isinstance(data, list):
            print(f"Error: The data in {file_path} is not a list.")
            return False
        
        # Iterate over the expected keys and validate if they exist
        for entry in data:
            for key, expected_type in expected_keys.items():
                if key not in entry:
                    print(f"Error: Key '{key}' not found in entry.")
                    return False

                # Check for the correct type for each expected key
                if isinstance(entry[key], list):  # Handle cases where values are lists
                    if not all(isinstance(i, expected_type) for i in entry[key]):
                        print(f"Error: Expected '{key}' to contain only {expected_type}, but got {type(entry[key][0])}.")
                        return False
                elif not isinstance(entry[key], expected_type):  # Check the type directly
                    print(f"Error: Expected '{key}' to be of type {expected_type}, but got {type(entry[key])}.")
                    return False
        
        # If we passed all checks, return True
        print(f"YAML file {file_path} passed validation.")
        return True
    
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return False

def validate_all_yaml_files(data_path, expected_keys):
    """
    Validate all YAML files in the specified data directory.
    
    Args:
        data_path (str): Path to the directory containing the YAML files.
        expected_keys (dict): A dictionary of expected keys and their types.
    """
    # Iterate through the files in the data directory
    for file_name in os.listdir(data_path):
        if file_name.endswith('.yaml'):
            file_path = os.path.join(data_path, file_name)
            is_valid = validate_yaml_structure(file_path, expected_keys)
            
            if not is_valid:
                print(f"Validation failed for {file_path}")
            else:
                print(f"{file_path} is valid.")

# Define the expected structure for your YAML files
expected_keys = {
    'obs': dict,  # 'obs' should contain a dictionary (e.g., 'pressure', 'pose', etc.)
    'pressure': float,
    'pose': dict,
    'distance': str,  # The distance could be a string with units in your example
    'twist_cmd': dict,  # The twist_cmd will likely be a nested dictionary
}

# Path to your dataset directory
data_path = '/home/grimmlins/IL_workspace/data_collection_output'

# Validate all YAML files in the data directory
validate_all_yaml_files(data_path, expected_keys)
