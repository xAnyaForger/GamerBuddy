import json
import nltk
import os
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from nltk.util import ngrams
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag

# Download required NLTK resources
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

def contains_non_english(text):
    return any(ord(c) > 127 for c in text)

# Load and prepare mapping from games.csv
games_df = pd.read_csv("datasets/games.csv")
id_to_title = dict(zip(games_df["app_id"], games_df["title"]))
games_info = games_df.set_index("app_id")[[
    "price_original", "price_final", "discount", "rating", "positive_ratio",
    "user_reviews", "win", "mac", "linux", "steam_deck"
]]

# Paths
input_path = "datasets/games_metadata.json"
output_dir = "datasets"
os.makedirs(output_dir, exist_ok=True)
output_file = os.path.join(output_dir, "data.json")

# Clean and build dataset
cleaned_data = []
total_records = 0
valid_records = 0

with open(input_path, "r", encoding="utf-8") as f:
    for line in f:
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue

        total_records += 1
        app_id = item.get("app_id")
        description = item.get("description", "").strip()
        tags = item.get("tags", [])

        if not app_id or not description or not isinstance(tags, list) or not tags:
            continue
        if contains_non_english(description) or any(contains_non_english(tag) for tag in tags):
            continue
        if app_id not in id_to_title or app_id not in games_info.index:
            continue

        meta = games_info.loc[app_id].to_dict()
        ordered_entry = {
            "app_id": app_id,
            "title": id_to_title[app_id],
            **meta,
            "description": description,
            "tags": tags
        }
        cleaned_data.append(ordered_entry)
        valid_records += 1

# Save cleaned JSON
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(cleaned_data, f, ensure_ascii=False, indent=2)

print("Cleaned data saved to:", output_file)
print("Total records before cleaning:", total_records)
print("Valid records after cleaning:", valid_records)

# NLP
corpus = " ".join([item["description"] for item in cleaned_data])
tokens = word_tokenize(corpus)
freq_dist = FreqDist(tokens)
bigrams = list(ngrams(tokens, 2))
trigrams = list(ngrams(tokens, 3))
lemmatizer = WordNetLemmatizer()
lemmas = [lemmatizer.lemmatize(token.lower()) for token in tokens if token.isalpha()]
pos_tags = pos_tag(tokens[:500])
vocabulary = sorted(set(token.lower() for token in tokens if token.isalpha()))

# Save NLP outputs
def save_list(path, data):
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(data))

save_list(os.path.join(output_dir, "tokens.txt"), tokens)

with open(os.path.join(output_dir, "frequency_distribution.txt"), "w", encoding="utf-8") as f:
    for word, freq in freq_dist.most_common():
        f.write(f"{word}\t{freq}\n")

with open(os.path.join(output_dir, "bigrams.txt"), "w", encoding="utf-8") as f:
    for bg in bigrams:
        f.write(" ".join(bg) + "\n")

with open(os.path.join(output_dir, "trigrams.txt"), "w", encoding="utf-8") as f:
    for tg in trigrams:
        f.write(" ".join(tg) + "\n")

save_list(os.path.join(output_dir, "lemmas.txt"), lemmas)
save_list(os.path.join(output_dir, "vocabulary.txt"), vocabulary)

with open(os.path.join(output_dir, "pos_tags.txt"), "w", encoding="utf-8") as f:
    for word, tag in pos_tags:
        f.write(f"{word}\t{tag}\n")

print(f"All NLP output files saved to folder: {output_dir}")
