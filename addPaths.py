import sys
import os

def add_all_subdirectories_to_path(base_dir):
    """
    Recursively adds all subdirectories within base_dir to sys.path.
    Warning: Use with caution. Can lead to module shadowing and import ambiguities.
    Consider Python's standard package structure with __init__.py files instead.

    Args:
        base_dir (str): The starting directory from which to find subdirectories.
    """
    if not os.path.isdir(base_dir):
        print(f"Error: '{base_dir}' is not a valid directory.", file=sys.stderr)
        return

    # Normalize the base directory path
    base_dir = os.path.abspath(base_dir)

    print(f"Adding subdirectories of '{base_dir}' to sys.path...")

    # Add the base directory itself first
    if base_dir not in sys.path:
        sys.path.insert(0, base_dir)
        print(f"  Added: {base_dir}")

    for root, dirs, files in os.walk(base_dir):
        # Add each directory found by os.walk to sys.path
        if root not in sys.path:
            sys.path.insert(0, root) # Insert at the beginning to give preference
            print(f"  Added: {root}")

    print("\nUpdated sys.path:")
    for p in sys.path[:10]: # Print first 10 for brevity
        print(f"  {p}")
    if len(sys.path) > 10:
        print("  ...")


if __name__ == "__main__":
    # Determine the base directory
    # Option 1: Use the directory where this script is located
    # base_directory_to_scan = os.path.dirname(os.path.abspath(__file__))

    # Option 2: Use the current working directory where you run the script from
    base_directory_to_scan = os.getcwd()

    # Option 3: Specify a hardcoded path (less flexible)
    # base_directory_to_scan = "/path/to/your/project/root"

    add_all_subdirectories_to_path(base_directory_to_scan)

    # --- Example Usage After Path Modification ---
    # Now you can try to import modules directly by their filename (without .py)
    # as long as they are in one of the added directories.
    # For instance, if you have a file 'my_module.py' in a subdirectory:
    # try:
    #     # This will only work if 'my_module.py' exists in one of the added paths
    #     # and doesn't conflict with built-in or installed packages.
    #     import my_module
    #     print("\nSuccessfully imported 'my_module' (example).")
    #     # Call a function from my_module if it exists
    #     # my_module.some_function()
    # except ImportError as e:
    #     print(f"\nCould not import 'my_module' (example): {e}")

    # You can also import specific classes or functions if you know their location
    # from some_subdir.another_module import MyClass