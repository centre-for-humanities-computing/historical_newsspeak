# %%
from datasets import load_dataset
import subprocess
import pandas as pd

tmp_dir = "../data/tmp_files"
out_dir = "../data/tagged_output/"

ROOT = "/Users/au324704/tagger_setup/"
TAGGER = f"{ROOT}taggerXML/taggerXML"

RESOURCE_DIR = f"{ROOT}texton-linguistic-resources/da/tagger/UTF8/c19/"
BIGRAMS = f"{RESOURCE_DIR}BIGBIGRAMLIST"
LEXRULE = f"{RESOURCE_DIR}LEXRULEOUTFILE"
CONTEXT = f"{RESOURCE_DIR}CONTEXT-RULEFILE"
LEXICON = f"{RESOURCE_DIR}FINAL.LEXICON"


# %%
ds = load_dataset("chcaa/press-and-plot", split="train")
# if you want it as a pandas DataFrame:
data = ds.to_pandas()
data.head()

# %%

number = data['text'].iloc[10]
print(number[:200])
example_text = number

# preprocess text, seperate out punctuation
example_text = example_text.replace(".", " . ").replace(",", " , ").replace("!", " ! ").replace("?", " ? ")

# Create a temporary file with the example text
with open(f"{tmp_dir}/temp_text.txt", "w") as tmp_file:
    tmp_file.write(example_text)

# Build the command pointing to the temp file
cmd = [TAGGER, LEXICON, tmp_file.name, BIGRAMS, LEXRULE, CONTEXT]

# Run the tagger
result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

if result.returncode != 0:
    print("Error:", result.stderr)
else:
    tagged_text = result.stdout
    print(tagged_text)

    # Parse output: assume whitespace separates tokens and / separate word and POS
    rows = []
    for line in tagged_text.strip().split("\n"):
        for token in line.split():
            if "/" in token:  # naive split; adjust to your tagger format
                word, pos = token.rsplit("/", 1)
                rows.append({"word": word, "pos": pos})

    # Create a DataFrame
    df = pd.DataFrame(rows)
    print(df.head)
    df.to_csv(f"{out_dir}/tagged_output.csv", index=False, encoding='utf-8-sig')

# %%

# now do the same with spacy
import spacy
nlp = spacy.load("da_core_news_sm")
doc = nlp(example_text)

spacy_rows = [{"word": token.text, "pos": token.pos_} for token in doc]
# remove spaces
spacy_rows = [row for row in spacy_rows if row["word"].strip() != ""]
spacy_df = pd.DataFrame(spacy_rows)
print(spacy_df.head)
spacy_df.to_csv(f"{out_dir}/spacy_tagged_output.csv", index=False, encoding='utf-8-sig')

# %%
# with dacy
import dacy

#open example txt
with open(f"text.txt", "r") as f:
    example_text = f.read()

# %%
dacy_model = dacy.load("da_dacy_large_trf-0.2.0")
dacy_doc = dacy_model(example_text)
#make df
dacy_rows = [{"word": token.text, "pos": token.pos_} for token in dacy_doc]
# remove spaces
dacy_rows = [row for row in dacy_rows if row["word"].strip() != ""]
dacy_df = pd.DataFrame(dacy_rows)
print(dacy_df.head)
dacy_df.to_csv(f"{out_dir}/dacy_tagged_output.csv", index=False, encoding='utf-8-sig')
# %%
dacy.models()
# %%

# 
from transformers import pipeline

# Load model and tokenizer
pos_pipeline = pipeline("token-classification", model="jordigonzm/mdeberta-v3-base-multilingual-pos-tagger")

# Input text
text = example_text

# Run POS tagging
words = text.split(" ")
tokens = pos_pipeline(words)

# Print tokens and their categories
for word, group_token in zip(words, tokens):
    print(f"{word:<15}", end=" ")
    for token in group_token:
        print(f"{token['word']:<8} → {token['entity']:<8}", end=" | ")
    print("\n" + "-" * 80)

# save to dataframe
transformers_rows = []
for word, group_token in zip(words, tokens):
    for token in group_token:
        transformers_rows.append({"word": word, "pos": token['entity']})
transformers_df = pd.DataFrame(transformers_rows)
# drop duplicates keeping first
transformers_df = transformers_df.drop_duplicates(subset=['word'], keep='first')
print(transformers_df.head)
transformers_df.to_csv(f"{out_dir}/transformers_tagged_output.csv", index=False, encoding='utf-8-sig')

# %%
