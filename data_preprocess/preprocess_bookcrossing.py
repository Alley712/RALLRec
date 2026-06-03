#!/usr/bin/env python3
"""
BookCrossing preprocessing — matches BookCrossing.ipynb behavior.
Two-pass: first 10-core filter to identify valid items, then build final dataset.
"""

import pandas as pd
import numpy as np
import json
import os
import re
import random
import copy
import hashlib
import h5py
from transformers import set_seed
from string import ascii_letters, digits
import torch
from torch.nn.utils.rnn import pad_sequence

set_seed(42)

# ============================================================
# Config
# ============================================================
dataset_name = "BookCrossing"
root = f"../data/{dataset_name}"
source_dir = os.path.join(root, "raw_data")
target_dir = os.path.join(root, "proc_data")
if not os.path.exists(target_dir):
    os.makedirs(target_dir)

# ============================================================
# Helpers
# ============================================================
def character_check(item, special_letters=""):
    for letter in str(item):
        if letter not in ascii_letters + digits + special_letters:
            return 1
    return 0

def isin_selected(item, selected_dict):
    return 1 if item in selected_dict else 0

# ============================================================
# 1. Read and process user info
# ============================================================
print("=" * 60)
print("Reading user info...")
print("=" * 60)

user_fields = ["User ID", "Location", "Age"]
pattern = re.compile(r'NULL|".*?(?<!\\)"', re.S)
with open(os.path.join(source_dir, "Book reviews/Book reviews/BX-Users.csv"), 'r', encoding='cp1252') as f:
    content = pattern.findall(f.read())
    content = [s[1:-1] if s != 'NULL' else None for s in content]
    processed_list = list(np.array(content).reshape((-1, 3)))
    processed_list.pop(0)
    df_users = pd.DataFrame(processed_list, columns=user_fields)

def convert_location_to_country(x):
    x = x.split(', ')[-1].strip().title().replace("!", "").strip()
    if x.lower() in ["usa", "us", "u s", "u s a"]:
        x = "USA"
    if x.lower() in ["uk", "u k"]:
        x = "UK"
    while len(x) > 0 and x[-1] in [",", "."]:
        x = x[:-1]
    while len(x) > 0 and x[0] in [",", "."]:
        x = x[1:]
    if "U.S" in x.upper() and x != "U.S. Virgin Islands":
        x = "USA"
    if x in ["San José", "San Josï¿½"]:
        x = "USA"
    if x in ["España", "Castilla-León", "Espaã±A", "Cataluña", "Mérida", "Álava", "Málaga",
              "A Coruña", "Barcelonès", "Berguedà", "Espaï¿½A", "Castilla-Leï¿½N",
              "A Coruï¿½A", "Cataluï¿½A", "Barcelonï¿½S", "Ï¿½Lava", "Mï¿½Rida",
              "Berguedï¿½", "Mï¿½Laga"] or "spain" in x.lower():
        x = "Spain"
    if x in ["L`Italia"]: x = "Italy"
    if x in ["Baden-Württemberg", "Bademn Würtemberg", "Baden-Wï¿½Rttemberg", "Bademn Wï¿½Rtemberg"]: x = "German"
    if x in ["Cote D`Ivoire", "Côte D", "Cï¿½Te D"]: x = "Ivory Coast"
    if x in ["Oberösterreich", "Oberï¿½Sterreich"]: x = "Austria"
    if x in ["México", "Mï¿½Xico"]: x = "Mexico"
    if x in ["Türkiye", "Içel", "Tï¿½Rkiye"]: x = "Turkey"
    if x in ["L`Algérie", "Algérie", "Kärnten", "Kï¿½Rnten", "L`Algï¿½Rie", "Algï¿½Rie"]: x = "Algeria"
    if "Brasil" in x: x = "Brazil"
    if x in ["Rhône-Alpes", "Rhône Alpes", "Rhï¿½Ne-Alpes", "Rhï¿½Ne Alpes"]: x = "France"
    if "Greece" in x: x = "Greece"
    if x in ["Santarém", "Santarï¿½M"]: x = "Portugal"
    if x in ["Länsi-Suomen Lääni", "Lï¿½Nsi-Suomen Lï¿½Ï¿½Ni"]: x = "Finland"
    if x in ["V.Götaland", "Nyhamnsläge", "V.Gï¿½Taland", "Nyhamnslï¿½Ge"]: x = "Sweden"
    if x in ["Moçambique", "Moï¿½Ambique"]: x = "Mozambique"
    if x in ["Ix Región", "Ix Regiï¿½N"]: x = "Chile"
    if x in ["Maï¿½Opolskie", "Ma³Opolskie"]: x = "Poland"
    if x in ["Perï¿½", "Perãº"]: x = "Peru"
    if x != "China" and ("china" in x.lower() or x == "La Chine Éternelle" or x == "La Chine Ï¿½Ternelle"): x = "China"
    if x == "Ï¿½Ï¿½Ï¿½": x = "China"
    if (x == "" or x in ["Öð¹Ú", "ºþäï", "We`Re Global", "Ï¿½Ï¿½Ï¿½Ï¿½", "Iï¿½El"] or
        len(x) == 1 or "N/A" in x or "&#" in x or "?" in x or "@" in x or "*" in x):
        x = "unknown"
    return x

df_users["Location"] = df_users["Location"].apply(convert_location_to_country)
df_users["location_check"] = df_users["Location"].apply(lambda x: character_check(x, special_letters="- .&/()"))
assert len(df_users.loc[df_users["location_check"] == 1, "Location"]) == 0

def convert_age_to_bucket(x):
    if x is None or (isinstance(x, float) and pd.isna(x)):
        x = "unknown"
    else:
        x = int(float(x))
        if x < 5 or x > 100: x = "unknown"
        elif x < 18: x = "under 18"
        elif 18 <= x < 25: x = "18-24"
        elif 25 <= x < 30: x = "25-29"
        elif 30 <= x < 35: x = "30-34"
        elif 35 <= x < 40: x = "35-39"
        elif 40 <= x < 45: x = "40-44"
        elif 45 <= x < 50: x = "45-49"
        elif 50 <= x < 55: x = "50-54"
        elif 55 <= x < 60: x = "55-59"
        else: x = "60+"
    return x

df_users["Age"] = df_users["Age"].apply(convert_age_to_bucket)
for field in user_fields:
    for s in list(df_users[field]):
        if field == "User ID": assert 1 <= int(s) <= 278858
        if field == "Location": assert 2 <= len(s) <= 45
        if field == "Age":
            assert s in ["unknown", "under 18","18-24","25-29","30-34","35-39",
                         "40-44","45-49","50-54","55-59","60+"]
df_users = df_users[user_fields]

md5 = hashlib.md5(json.dumps(df_users.values.tolist(), sort_keys=True).encode('utf-8')).hexdigest()
print(f"df_users md5: {md5}")
assert md5 == "111bda80ee793f1efcaf0f58cb920771"
print(f"Users loaded: {len(df_users)}")

# ============================================================
# 2. Read and process book info
# ============================================================
print("\nReading book info...")
book_fields = ["ISBN", "Book title", "Author", "Publication year", "Publisher"]
pattern = re.compile(r'(?<=");(?=")')
processed_list = []
with open(os.path.join(source_dir, "Book reviews/Book reviews/BX_Books.csv"), 'r', encoding='cp1252') as f:
    for line in f.readlines():
        split_line = pattern.split(line.strip())
        split_line = [item[1:-1].strip('\t') for item in split_line][:-3]
        processed_list.append(split_line)
    processed_list.pop(0)
    df_books = pd.DataFrame(processed_list, columns=book_fields)

df_books['ISBN_check'] = df_books['ISBN'].apply(lambda x: character_check(x))
df_books = df_books[df_books['ISBN_check'] == 0]

def convert_publication_year(x):
    return x if len(x) == 4 else "unknown"

df_books["Publication year"] = df_books["Publication year"].apply(convert_publication_year)
df_books["Publisher"] = df_books["Publisher"].apply(lambda x: x if x.lower() != "n/a" else "unknown")
df_books["Author"] = df_books["Author"].apply(lambda x: x if x.lower() != "n/a" else "unknown")

for field in book_fields:
    for s in list(df_books[field]):
        if field == "ISBN": assert len(s) == 10
        if field == "Book title": assert 1 <= len(s) <= 256
        if field == "Author": assert 1 <= len(s) <= 143
        if field == "Publication year": assert s == "unknown" or len(s) == 4
        if field == "Publisher": assert 1 <= len(s) <= 134

df_books = df_books[book_fields]
print(f"Books loaded: {len(df_books)}")

# Build isbn→info map
all_isbn_info = {}
for idx, row in df_books.iterrows():
    all_isbn_info[row["ISBN"]] = [row["Book title"], row["Author"],
                                   row["Publication year"], row["Publisher"]]

# ============================================================
# 3. Read ratings (UNFILTERED)
# ============================================================
print("\nReading ratings...")
processed_list = []
with open(os.path.join(source_dir, "Book reviews/Book reviews/BX-Book-Ratings.csv"), 'r', encoding='cp1252') as f:
    for line in f.readlines():
        split_line = line.strip().split(';')
        split_line = [item[1:-1] for item in split_line]
        processed_list.append(split_line)
    processed_list.pop(0)

# ============================================================
# 4. Apply 10-core filtering to identify valid items (matches commented notebook cells)
# ============================================================
print("\nApplying 10-core filtering to identify valid items...")

# First, read all ratings into a dataframe for filtering
all_ratings = pd.DataFrame(processed_list, columns=["User ID", "ISBN", "Rating"])
print(f"Total ratings: {len(all_ratings)}")

# 10-core filter: items >= 10, users > 10 and <= 200
def filter_10_core(data, user_col, item_col):
    while True:
        user_counts = data[user_col].value_counts()
        valid_users = user_counts[(user_counts > 10) & (user_counts <= 200)].index
        data = data[data[user_col].isin(valid_users)]

        item_counts = data[item_col].value_counts()
        valid_items = item_counts[item_counts >= 10].index
        data = data[data[item_col].isin(valid_items)]

        if len(valid_users) == len(user_counts) and len(valid_items) == len(item_counts):
            break
    return data

filtered_df = filter_10_core(all_ratings, user_col='User ID', item_col='ISBN')
print(f"10-core filtered: {len(filtered_df)} ratings, {filtered_df['User ID'].nunique()} users, {filtered_df['ISBN'].nunique()} items")

# Build ibsn_selected from 10-core filtered items
# Only keep items that also have book metadata
valid_isbns = set(filtered_df['ISBN'].unique()) & set(all_isbn_info.keys())
print(f"Valid ISBNs (in book metadata): {len(valid_isbns)}")

ibsn_selected = {isbn: 1 for isbn in valid_isbns}

# Save isbn2id at this point (this is what the notebook's cell 6 loads)
isbn_list = sorted(valid_isbns)
isbn2id = {isbn: i for i, isbn in enumerate(isbn_list)}
id2book = {str(i): [isbn] + all_isbn_info[isbn] for isbn, i in isbn2id.items()}
print(f"isbn2id: {len(isbn2id)} entries")

# ============================================================
# 5. Build user histories from FULL ratings, filtered by valid ISBNs
# ============================================================
print("\nBuilding user histories (full ratings, filtered by valid ISBNs)...")
user_hist, hist_rating, labels = {}, {}, {}
for user, isbn, rating in processed_list:
    if isbn in ibsn_selected:
        if user not in user_hist:
            user_hist[user] = []
            hist_rating[user] = []
            labels[user] = []
        user_hist[user].append(isbn)
        hist_rating[user].append(int(rating))
        labels[user].append(int(int(rating) >= 5))

print(f"Users with any valid rating: {len(user_hist)}")

# Filter users with < 5 ratings
user_del = [u for u, h in user_hist.items() if len(h) < 5]
print(f"Removing {len(user_del)} users with < 5 ratings")
for u in user_del:
    del user_hist[u]; del hist_rating[u]; del labels[u]
print(f"Users after 5-core: {len(user_hist)}")

# Build user_selected
user_selected = {user: 1 for user in user_hist}

# ============================================================
# 6. Filter users & books dataframes using selection dicts
# ============================================================
print("\nFiltering user/book dataframes...")
df_users['check'] = df_users['User ID'].apply(lambda x: isin_selected(x, user_selected))
df_users = df_users[df_users['check'] == 1]
df_users = df_users[user_fields]
print(f"Filtered users: {len(df_users)}")

df_books['ISBN_check'] = df_books['ISBN'].apply(lambda x: isin_selected(x, ibsn_selected))
df_books = df_books[df_books['ISBN_check'] == 1]
df_books = df_books[book_fields]
print(f"Filtered books: {len(df_books)}")

# Verify consistency
assert len(df_users) == len(user_selected), f"{len(df_users)} != {len(user_selected)}"
assert len(df_books) == len(ibsn_selected), f"{len(df_books)} != {len(ibsn_selected)}"

# Build feature_dict and user/book dicts
print("Building feature dictionaries...")
def add_to_dict(d, feature):
    if feature not in d:
        d[feature] = len(d)

feature_dict = {field: {} for field in user_fields + book_fields}
user_dict = {}
book_dict = {}

for idx, row in df_users.iterrows():
    if row["User ID"] not in user_dict:
        user_dict[row["User ID"]] = [row["Location"], row["Age"]]
    for field in user_fields:
        add_to_dict(feature_dict[field], row[field])

for idx, row in df_books.iterrows():
    if row["ISBN"] not in book_dict:
        book_dict[row["ISBN"]] = [row["Book title"], row["Author"], row["Publication year"], row["Publisher"]]
    for field in book_fields:
        add_to_dict(feature_dict[field], row[field])

feature_count = [len(feature_dict[field]) for field in user_fields + book_fields]
for field in user_fields + book_fields:
    print(f"  {field}: {len(feature_dict[field])}")

# Verify feature dicts
for field in user_fields:
    assert len(feature_dict[field]) == len(set(list(df_users[field])))
for field in book_fields:
    assert len(feature_dict[field]) == len(set(list(df_books[field])))

# ============================================================
# 7. Shuffle histories
# ============================================================
print("\nShuffling histories...")
for user in user_hist.keys():
    zipped = list(zip(user_hist[user], hist_rating[user], labels[user]))
    set_seed(42)
    random.shuffle(zipped)
    user_hist[user], hist_rating[user], labels[user] = map(list, zip(*zipped))

# ============================================================
# 8. Build data_list
# ============================================================
print("Building data_list...")
data_list = []
for user in user_hist.keys():
    isbn = user_hist[user][-1]  # last item = target
    data_sample = copy.deepcopy(
        [user] + user_dict[user] + [isbn] + book_dict[isbn] +
        [user_hist[user][:-1]] + [hist_rating[user][:-1]] + [labels[user][-1]] + [hist_rating[user][-1]]
    )
    data_list.append(data_sample)

print(f"data_list: {len(data_list)} samples")

vals = [labels[u][-1] for u in user_hist.keys()]
print(f"Label dist - max: {max(vals)}, mean: {sum(vals)/len(vals):.4f}")

# ============================================================
# 9. Shuffle and split
# ============================================================
print("\nShuffling and splitting...")
set_seed(42)
random.shuffle(data_list)
df_data = pd.DataFrame(data_list, columns=user_fields + book_fields + ["user_hist", "hist_rating", "labels", "rating"])
print(f"Total samples: {len(df_data)}")

df_train = df_data[:int(0.9 * len(df_data))].reset_index(drop=True)
df_test = df_data[int(0.9 * len(df_data)):].reset_index(drop=True)
print(f"Train: {len(df_train)}, Test: {len(df_test)}")

# Save parquet
df_train.to_parquet(os.path.join(target_dir, "train.parquet.gz"), compression="gzip")
df_test.to_parquet(os.path.join(target_dir, "test.parquet.gz"), compression="gzip")
print("Parquet files saved.")

# Verify parquet
train_ds = pd.read_parquet(os.path.join(target_dir, "train.parquet.gz"))
test_ds = pd.read_parquet(os.path.join(target_dir, "test.parquet.gz"))
for (i1, a1), (i2, a2) in zip(df_train.iterrows(), train_ds.iterrows()):
    for field in user_fields + book_fields + ["labels"]:
        assert not isinstance(a1[field], str) or "\t" not in a1[field]
        assert a1[field] == a2[field], (field, a1[field], a2[field])
for (i1, a1), (i2, a2) in zip(df_test.iterrows(), test_ds.iterrows()):
    for field in user_fields + book_fields + ["labels"]:
        assert not isinstance(a1[field], str) or "\t" not in a1[field]
        assert a1[field] == a2[field], (field, a1[field], a2[field])
print("Parquet verification passed.")

# ============================================================
# 10. Save CTR metadata, id2book, isbn2id
# ============================================================
print("\nSaving metadata...")
field_names = user_fields + book_fields
feature_offset = [0]
for c in feature_count[:-1]:
    feature_offset.append(feature_offset[-1] + c)

meta_data = {
    'field_names': field_names,
    'feature_count': feature_count,
    'feature_dict': feature_dict,
    'feature_offset': feature_offset,
    'num_ratings': 11
}
json.dump(meta_data, open(os.path.join(target_dir, 'ctr-meta.json'), 'w'))

# Rebuild isbn2id and id2book from feature_dict (consistent with notebook cell 33)
isbn2id_final = meta_data['feature_dict']['ISBN']
id2book_final = {str(book_id): [isbn] + book_dict[isbn] for isbn, book_id in isbn2id_final.items()}
json.dump(id2book_final, open(os.path.join(target_dir, 'id2book.json'), "w"), indent=4)
json.dump(isbn2id_final, open(os.path.join(target_dir, 'isbn2id.json'), "w"), indent=4)
json.dump(book_dict, open(os.path.join(target_dir, "book_dict.json"), "w"), indent=4)
print(f"id2book: {len(id2book_final)}, isbn2id: {len(isbn2id_final)}")

# ============================================================
# 11. Build and save CTR data
# ============================================================
print("\nBuilding CTR data...")
ctr_X, ctr_Y = [], []
for idx, row in df_data.iterrows():
    ctr_X.append([feature_dict[field][row[field]] for field in field_names])
    ctr_Y.append(int(row["labels"]))

ctr_X = np.array(ctr_X)
ctr_Y = np.array(ctr_Y)
print(f"ctr_X: {ctr_X.shape}, ctr_Y: {ctr_Y.shape}")
assert (ctr_X - np.array(feature_count).reshape(1, -1) <= 0).sum() == ctr_X.shape[0] * ctr_X.shape[1]

# User sequences
history_column = {}
history_column["ID"] = [[isbn2id_final[x] for x in hist] for hist in df_data['user_hist'].tolist()]
history_column["rating"] = df_data['hist_rating'].tolist()
history_column["hist length"] = [len(x) for x in history_column["rating"]]

train_num = int(0.9 * len(ctr_X))

user_seq = {
    "history ID": {"train": history_column["ID"][:train_num], "test": history_column["ID"][train_num:]},
    "history rating": {"train": history_column["rating"][:train_num], "test": history_column["rating"][train_num:]},
    "history length": {"train": history_column["hist length"][:train_num], "test": history_column["hist length"][train_num:]},
}
json.dump(user_seq, open(os.path.join(target_dir, "user_seq.json"), "w"), ensure_ascii=False)

# Truncate and pad
user_seq_trunc = {"history ID": {}, "history rating": {}, "history mask": {}}
for hist_name in user_seq:
    for split in user_seq[hist_name]:
        if hist_name != "history length":
            user_seq_trunc[hist_name][split] = pad_sequence(
                [torch.tensor(x[-60:]) for x in user_seq[hist_name][split]], batch_first=True)
        else:
            user_seq_trunc["history mask"][split] = pad_sequence(
                [torch.ones(min(x, 60)) for x in user_seq[hist_name][split]], batch_first=True)

for hist_name in user_seq_trunc:
    for split in user_seq_trunc[hist_name]:
        print(f"  {hist_name} {split}: {user_seq_trunc[hist_name][split].shape}")

# Save h5
print("Saving CTR h5...")
with h5py.File(os.path.join(target_dir, 'ctr.h5'), 'w') as hf:
    hf.create_dataset('train data', data=ctr_X[:train_num, :])
    hf.create_dataset('test data', data=ctr_X[train_num:, :])
    hf.create_dataset('train label', data=ctr_Y[:train_num])
    hf.create_dataset('test label', data=ctr_Y[train_num:])
    for hist_name in user_seq_trunc:
        for split in user_seq_trunc[hist_name]:
            hf.create_dataset(f"{split} {hist_name}", data=user_seq_trunc[hist_name][split])

# Quick verify
with h5py.File(os.path.join(target_dir, 'ctr.h5'), 'r') as hf:
    assert (ctr_X - np.concatenate([hf['train data'][:], hf['test data'][:]], axis=0)).sum() == 0
    assert (ctr_Y - np.concatenate([hf['train label'][:], hf['test label'][:]], axis=0)).sum() == 0
    for hist_name in user_seq_trunc:
        for split in user_seq_trunc[hist_name]:
            assert (user_seq_trunc[hist_name][split] - hf[f"{split} {hist_name}"][:]).sum() == 0
print("CTR h5 verified.")

# Final cross-check
train_ds = pd.read_parquet(os.path.join(target_dir, 'train.parquet.gz'))
test_ds = pd.read_parquet(os.path.join(target_dir, 'test.parquet.gz')).reset_index(drop=True)
with h5py.File(os.path.join(target_dir, 'ctr.h5'), 'r') as hf:
    train_x, train_y = hf['train data'][:], hf['train label'][:]
    test_x, test_y = hf['test data'][:], hf['test label'][:]
for idx, row in train_ds.iterrows():
    for fi, field in enumerate(field_names):
        assert feature_dict[field][row[field]] == train_x[idx, fi]
    assert int(row["labels"]) == train_y[idx]
for idx, row in test_ds.iterrows():
    for fi, field in enumerate(field_names):
        assert feature_dict[field][row[field]] == test_x[idx, fi]
    assert int(row["labels"]) == test_y[idx]
print("Final verification passed!")

# ============================================================
# 12. Build LightGCN format train/test txt
# ============================================================
print("\nBuilding LightGCN train/test txt files...")
rating_data = []
for user, isbn, rating in processed_list:
    if user in feature_dict["User ID"] and isbn in feature_dict["ISBN"]:
        rating_data.append([user, isbn, rating])

df_ratings = pd.DataFrame(rating_data, columns=["User ID", "ISBN", "rating"])
print(f"Rating data: {df_ratings.shape}")

# 5-core filter for recommendations
book_counts = df_ratings['ISBN'].value_counts()
valid_books = book_counts[book_counts >= 5].index
print(f"Books with >= 5 ratings: {len(valid_books)}")

user_groups = df_ratings.groupby('User ID').filter(lambda x: len(x) > 5)
tail_set = user_groups.groupby('User ID').tail(1)
test_set = tail_set[tail_set['ISBN'].isin(valid_books)]
train_set = df_ratings.drop(test_set.index)
print(f"Train: {train_set.shape}, Test: {test_set.shape}")

df_train_rec = train_set.sort_values(by=["User ID", "ISBN"], inplace=False, kind="stable")
df_test_rec = test_set.sort_values(by=["User ID", "ISBN"], inplace=False, kind="stable")
df_train_rec.to_csv(os.path.join(target_dir, "train.txt"), sep=' ', index=False, header=None)
df_test_rec.to_csv(os.path.join(target_dir, "test.txt"), sep=' ', index=False, header=None)

# ============================================================
# Done
# ============================================================
print("\n" + "=" * 60)
print("Preprocessing complete!")
print("=" * 60)
for f in sorted(os.listdir(target_dir)):
    fpath = os.path.join(target_dir, f)
    size = os.path.getsize(fpath) if os.path.isfile(fpath) else "N/A"
    print(f"  {f}: {size:,} bytes")
