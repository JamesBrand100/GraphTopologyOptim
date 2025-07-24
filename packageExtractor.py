import os

def extract_packages(file_path):

    if not os.path.exists(file_path):
        print(f"Error: File not found at '{file_path}'")
        return

    print(f"Searching for r'usepackage' lines in '{file_path}'...\n")
    found_packages = False
    try:
        # Using 'latin-1' encoding which maps every byte to a Unicode code point,
        # ensuring that no UnicodeDecodeError occurs for arbitrary byte sequences.
        with open(file_path, 'r', encoding='latin-1') as f:
            for line_num, line in enumerate(f, 1):
                # Check for '\usepackage' (case-insensitive for robustness, though usually lowercase)
                # and ignore commented out lines (starting with %)
                stripped_line = line.strip()
                if stripped_line.startswith('%'):
                    continue
                if 'usepackage' in stripped_line:
                    print(f"Line {line_num}: {stripped_line}")
                    found_packages = True
        if not found_packages:
            print("No r'usepackage' lines found in the file.")
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")

# --- How to use the script ---
# Replace 'your_document.txt' with the actual path to your text file.
# Example:
# text_file = 'my_notes.txt'
# text_file = '/path/to/your/document/chapter_notes.txt'

# IMPORTANT: Replace the placeholder below with the actual path to your text file.
# For local testing, ensure the file is in the same directory as the script,
# or provide its full path.
latex_file_path = 'packageList.txt' # <--- !!! CHANGE THIS TO YOUR FILE PATH !!!

# Call the function to extract packages
extract_packages(latex_file_path)
