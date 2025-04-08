import spacy
import os
import glob

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

def extract_nouns_from_line(line):
    doc = nlp(line)
    return [token.text for token in doc if token.pos_ == "NOUN"]

def process_all_train_files(folder_path):
    noun_set = set()
    # Find all files ending with "train.txt"
    file_list = glob.glob(os.path.join(folder_path, "*train.txt"))
    
    for file_path in file_list:
        print(f"Processing: {file_path}")
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                nouns = extract_nouns_from_line(line.strip())
                noun_set.update(noun.lower() for noun in nouns)  # lowercase to unify

    return sorted(noun_set)

# Example usage
folder_path = '/home/dohyun/kdh/text-aligned_img/T2I-CompBench/examples/dataset/'
unique_nouns = process_all_train_files(folder_path)

print(f"\nTotal unique nouns: {len(unique_nouns)}")
print(unique_nouns)
print("num_noun", len(unique_nouns))