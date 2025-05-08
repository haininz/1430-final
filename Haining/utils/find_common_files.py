import os

def find_common_basenames(file_paths, output_filename):
    file_sets = []
    for file_path in file_paths:
        with open(file_path, 'r') as f:
            basenames = {os.path.basename(line.strip()) for line in f if line.strip()}
            print(f"{file_path}: {len(basenames)} unique basenames loaded")
            file_sets.append(basenames)

    if not file_sets:
        print(f"\nNo data found in {output_filename}")
        return set()

    common_files = set.intersection(*file_sets)

    print(f"\nFound {len(common_files)} common basenames in {output_filename}:")
    for fname in sorted(common_files):
        print(fname)

    os.makedirs("../../Haining/results", exist_ok=True)
    output_path = os.path.join("../../Haining/results", output_filename)

    with open(output_path, 'w') as out:
        for fname in sorted(common_files):
            out.write(f"{fname}\n")

    print(f"\nCommon basename list saved to {output_path}\n")
    return common_files


files_ai_correctly_classified = [
    "../../Haining/data/ai_correctly_classified.txt",
    "../../Brayden/ai_correctly_classified.txt",
    "../../Marina/ai_correctly_classified.txt",
    "../../Nadia/ai_correctly_classified.txt"
]

files_natural_correctly_classified = [
    "../../Haining/data/natural_correctly_classified.txt",
    "../../Brayden/natural_correctly_classified.txt",
    "../../Marina/natural_correctly_classified.txt",
    "../../Nadia/natural_correctly_classified.txt"
]

files_natural_misclassified_as_ai = [
    "../../Haining/data/natural_misclassified_as_ai.txt",
    "../../Brayden/natural_misclassified_as_ai.txt",
    "../../Marina/natural_misclassified_as_ai.txt",
    "../../Nadia/natural_misclassified_as_ai.txt"
]

files_ai_misclassified_as_natural = [
    "../../Haining/data/ai_misclassified_as_natural.txt",
    "../../Brayden/ai_misclassified_as_natural.txt",
    "../../Marina/ai_misclassified_as_natural.txt",
    "../../Nadia/ai_misclassified_as_natural.txt"
]


find_common_basenames(files_ai_correctly_classified, "common_ai_correctly_classified.txt")
find_common_basenames(files_natural_correctly_classified, "common_natural_correctly_classified.txt")
find_common_basenames(files_natural_misclassified_as_ai, "common_natural_misclassified_as_ai.txt")
find_common_basenames(files_ai_misclassified_as_natural, "common_ai_misclassified_as_natural.txt")