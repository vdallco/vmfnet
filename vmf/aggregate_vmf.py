import os

def collect_raw_vmf_files(input_folder, output_file):
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for root, _, files in os.walk(input_folder):
            for file in files:
                if file.endswith('.vmf'):
                    file_path = os.path.join(root, file)
                    print("Processing file:", file_path)
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as infile:
                        content = infile.read()
                        outfile.write(content.strip() + "\n\n")  # Add spacing between files

# Example usage
collect_raw_vmf_files("./vmf/output", "vmf_raw_corpus.txt")