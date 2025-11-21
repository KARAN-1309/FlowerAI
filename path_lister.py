import os

def get_all_file_paths(directory_path):
    """
    Recursively traverses a directory and returns a list of all file paths.
    
    Args:
        directory_path (str): The path to the root directory to search.
        
    Returns:
        list: A list of string paths for every file found.
    """
    # List to store the paths of all files
    file_paths = []

    # Verify the directory exists before starting
    if not os.path.exists(directory_path):
        print(f"Error: The directory '{directory_path}' does not exist.")
        return []

    # os.walk() generates the file names in a directory tree
    # by walking the tree either top-down or bottom-up.
    for root, directories, files in os.walk(directory_path):
        for filename in files:
            # Join the two strings in order to form the full filepath.
            # root is the current directory path being traversed
            filepath = os.path.join(root, filename)
            file_paths.append(filepath)

    return file_paths

if __name__ == "__main__":
    # define the directory name
    # Use '.' to search the current directory where the script is running
    target_directory = '.'
    
    # Get the paths
    # Print the absolute path so we know exactly where we are looking
    full_path = os.path.abspath(target_directory)
    print(f"Scanning '{full_path}' for files...\n")
    
    paths = get_all_file_paths(target_directory)
    
    # Output filename
    output_file = "file_paths.txt"

    # Print the results and write to file
    if paths:
        with open(output_file, "w", encoding="utf-8") as f:
            for path in paths:
                print(path)
                f.write(path + "\n")
        
        print(f"\nTotal files found: {len(paths)}")
        print(f"List of paths saved to '{output_file}'")
    else:
        print("No files found or directory does not exist.")