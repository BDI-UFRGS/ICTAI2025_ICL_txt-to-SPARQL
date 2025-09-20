import datetime
import re

def formatted_timestamp():
    return datetime.datetime.now().strftime('%Y-%b-%d %H:%M:%S.%f')
    
def generate_log(message: str):
    return f"{formatted_timestamp()}: {message}"

def generate_note(message: str):
    return f"{formatted_timestamp()}: {message}"

def extract_timestamp_of(message: str):
    timestamp_pattern = r"(\d{4}-[A-Za-z]{3}-\d{2} \d{2}:\d{2}:\d{2}\.\d{6})"
    match = re.search(timestamp_pattern, message)
    if match:
        return match.group(1)
    else:
        print("The parameter provided to 'extract_timestamp_of()' does not contain a timestamp pattern")