from transformers import GPT2Tokenizer
import os

def collect_and_tokenize_vmf_files(input_folder, output_file):
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")  # Load GPT-2 tokenizer

    with open(output_file, 'w', encoding='utf-8') as outfile:
        for root, _, files in os.walk(input_folder):
            for file in files:
                if file.endswith('.vmf'):
                    file_path = os.path.join(root, file)
                    print("Processing file:", file_path)
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as infile:
                        # Read and tokenize each file
                        content = infile.read()
                        tokenized_content = tokenizer.encode(content, add_special_tokens=False)  # Tokenize
                        print("Tokenized content:", tokenized_content)
                        # Write the tokens to the output file
                        outfile.write(' '.join(map(str, tokenized_content)) + "\n\n")  # Separate each file with newlines

# Example usage
collect_and_tokenize_vmf_files("./vmf/output", "vmf_tokenized_corpus.txt")
