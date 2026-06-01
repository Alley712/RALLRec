#!/usr/bin/env python
# coding: utf-8

# In[18]:


import pandas as pd
import numpy as np
import json
import ast
import os
import re
import random
import copy
from transformers import set_seed
import hashlib
import json
import pickle as pkl
import h5py
import collections
from tqdm import tqdm

set_seed(42)

dataset_name = "amazon-movies"
root = f"../data/{dataset_name}"
source_dir = os.path.join(root, "raw_data")
target_dir = os.path.join(root, "proc_data")

os.makedirs(target_dir, exist_ok=True)


# In[19]:


obj = []
with open(os.path.join(source_dir, "meta_Movies_and_TV.json"), 'r') as fp:
    for line in fp:
        ele = ast.literal_eval(line.strip())
        obj.append(ele)


# In[36]:


print(len(obj))
print(obj[1])


# In[37]:


obj1 = []
cnt = 0
with open(os.path.join(source_dir, "Movies_and_TV.json"), 'r') as fp:
    for line in fp:
        cnt += 1
        ele = json.loads(line.strip())
        obj1.append(ele)
        if cnt >= 5:
            break


# In[40]:


print(obj1[1])


# In[41]:


# Movie data

movie_data = []
movie_detail = {}
movie_fields = ["Movie ID", "Movie title", "Movie category"]
with open(os.path.join(source_dir, "meta_Movies_and_TV.json"), 'r') as fp:
    for line in fp:
        ele = ast.literal_eval(line.strip())
        if "asin" not in ele or "title" not in ele:
            continue
        movie_id = ele["asin"].strip()
        if not movie_id:
            continue
        movie_title = ele["title"].strip()
        # Old UCSD format: categories is list of lists, e.g. [['Movies & TV', 'Comedy']]
        cats = ele.get("categories", [])
        if cats and len(cats[0]) >= 1:
            movie_genre = cats[0][-1]  # most specific category
        else:
            movie_genre = "unknown"
        movie_data.append([movie_id, movie_title, movie_genre])
        movie_detail[movie_id] = [movie_title, movie_genre]

df_movie = pd.DataFrame(movie_data, columns=movie_fields)
print(f"Total number of movies: {len(df_movie)}")

json.dump(movie_detail, open(os.path.join(target_dir, "movie_detail.json"), "w"))


# In[44]:


movie_dict = {}
movie_list = list(df_movie["Movie ID"])
for id in movie_list:
    movie_dict[id] = 1


# In[45]:


# Rating data
from datetime import datetime

rating_data = []
rating_fields = ["User ID", "Movie ID", "rating", "timestamp", "labels"]
with open(os.path.join(source_dir, "Movies_and_TV.json"), 'r') as fp:
    for line in fp:
        ele = json.loads(line.strip())
        user_id = ele["reviewerID"].strip()
        movie_id = ele["asin"].strip()
        rating = int(ele["overall"])
        timestamp = int(datetime.strptime(ele["reviewTime"], "%m %d, %Y").timestamp())
        label = 1 if rating > 3 else 0
        if movie_id in movie_dict:
            rating_data.append([user_id, movie_id, rating, timestamp, label])

df_ratings = pd.DataFrame(rating_data, columns=rating_fields)
print(f"Total number of ratings: {len(df_ratings)}")


# In[46]:


# Merge df_user/df_movie/df_rating into df_data

df_data = pd.merge(df_ratings, df_movie, on=["Movie ID"], how="inner")

df_data = df_data[df_data["Movie category"] != "unknown"]

df_data.sort_values(by=["timestamp", "User ID", "Movie ID"], inplace=True, kind="stable")

field_names = ["timestamp", "User ID", "Movie ID", "Movie title", "Movie category", "rating", "labels"]

df_data = df_data[field_names].reset_index(drop=True)

df_data.head()


# In[102]:


# NOTE: original cell "len(filtered_df)" removed — filtered_df not defined at this point


# In[93]:


import pandas as pd

def filter_10_core(data, user_col, item_col):
    """
    Iteratively filters the dataset to ensure every user and item has at least 10 interactions.
    
    :param data: The raw dataset as a Pandas DataFrame.
    :param user_col: Column name for users.
    :param item_col: Column name for items.
    :return: Filtered DataFrame where each user and item has at least 10 interactions.
    """
    while True:

        # Filter users with at least 10 history interactions but no more than 200
        user_counts = data[user_col].value_counts()
        valid_users = user_counts[(user_counts > 10)&(user_counts <= 200)].index
        data = data[data[user_col].isin(valid_users)]

        # Filter items with at least 10 interactions
        item_counts = data[item_col].value_counts()
        valid_items = item_counts[(item_counts >= 10)&(item_counts <= 200)].index
        data = data[data[item_col].isin(valid_items)]
        
        # Check if the dataset is stable (no more filtering needed)
        if len(valid_users) == len(user_counts) and len(valid_items) == len(item_counts):
            break

    return data


# Example usage:
# Assuming you have a dataset `df` with columns 'user_id' and 'movie_id'
filtered_df = filter_10_core(df_data, user_col='User ID', item_col='Movie ID')

# Save the filtered dataset
filtered_df.to_csv(os.path.join(source_dir, "amazon_movies_10_core.csv"), index=False)


# In[116]:


movie_select = {}
movie_list = list(filtered_df["Movie ID"])
item_counts = filtered_df["Movie ID"].value_counts()
movie_list_subset = item_counts.index[:5000]
for id in movie_list_subset:
    movie_select[id] = 1


# In[118]:


df_data_filtered = filtered_df[filtered_df["Movie ID"].isin(movie_select)]


# In[135]:


len(df_data_filtered)


# In[144]:


df_data = df_data_filtered.reset_index(drop=True)


# In[149]:


len(df_data)


# In[146]:


# Collect user history (<= 30)

user_history_dict = {
    "ID": {k: [] for k in set(df_data["User ID"])},
    "rating": {k: [] for k in set(df_data["User ID"])},
}
history_column = {
    "ID": [],
    "rating": [],
}
movie_id_to_title = {}

for idx, row in tqdm(df_data.iterrows()):
    user_id, movie_id, rating, title = row["User ID"], row["Movie ID"], row["rating"], row["Movie title"]
    history_column["ID"].append(user_history_dict["ID"][user_id].copy())
    history_column["rating"].append(user_history_dict["rating"][user_id].copy())
    user_history_dict["ID"][user_id].append(movie_id)
    user_history_dict["rating"][user_id].append(rating)
    if movie_id not in movie_id_to_title:
        movie_id_to_title[movie_id] = title

json.dump(movie_id_to_title, open(os.path.join(target_dir, "id_to_title.json"), "w"))


# In[147]:


# Drop data sample with history length that is less than 5.

df_data["history ID"] = history_column["ID"]
df_data["history rating"] = history_column["rating"]

df_data = df_data[df_data["history ID"].apply(lambda x: len(x)) >= 5].reset_index(drop=True)

history_column["ID"] = [x for x in history_column["ID"] if len(x) >= 5]
history_column["rating"] = [x for x in history_column["rating"] if len(x) >= 5]
history_column["hist length"] = [len(x) for x in history_column["rating"]]

for idx, row in tqdm(df_data.iterrows()):
    assert row["history ID"] == history_column["ID"][idx]
    assert row["history rating"] == history_column["rating"][idx]
    assert len(row["history rating"]) == history_column["hist length"][idx]


print(df_data.head())

print(f"Number of data sampels: {len(df_data)}")


# In[158]:


# Encode the feature dict for CTR data

def add_to_dict(dict, feature):
    if feature not in dict:
        dict[feature] = len(dict)

field_names = ["User ID", "Movie ID", "Movie title", "Movie category"]
feature_dict = {field : {} for field in field_names}


for idx, row in tqdm(df_data.iterrows()):
    for field in field_names:
        add_to_dict(feature_dict[field], row[field])

feature_count = [len(feature_dict[field]) for field in field_names]

feature_offset = [0]
for c in feature_count[:-1]:
    feature_offset.append(feature_offset[-1] + c)

for field in field_names:
    print(field, len(feature_dict[field]))

print("---------------------------------------------------------------")
for f, fc, fo in zip(field_names, feature_count, feature_offset):
    print(f, fc, fo)
print("---------------------------------------------------------------")


# In[159]:


# Save the meta data for CTR

meta_data = {
    "field_names": field_names,
    "feature_count": feature_count,
    "feature_dict": feature_dict,
    "feature_offset": feature_offset,
    "movie_id_to_title": movie_id_to_title,
    "num_ratings": 5,
}


json.dump(meta_data, open(os.path.join(target_dir, "ctr-meta.json"), "w"), ensure_ascii=False)


# In[160]:


movie_dict = json.load(open(os.path.join(target_dir, 'movie_detail.json')))
meta_data = json.load(open(os.path.join(target_dir, 'ctr-meta.json')))
id2idx = meta_data['feature_dict']['Movie ID']
idx2movie = {idx: [movie_id] + movie_dict[movie_id] for movie_id, idx in id2idx.items()}
json.dump(idx2movie, open(os.path.join(target_dir, 'idx2movie.json'), "w"), indent=4)
json.dump(id2idx, open(os.path.join(target_dir, 'id2idx.json'), "w"), indent=4)


# In[161]:


# Split & save user history sequence

train_num = int(0.8 * len(df_data))
valid_num = int(0.1 * len(df_data))
test_num = len(df_data) - train_num - valid_num

history_column["ID"] = [[id2idx[x] for x in hist] for hist in df_data['history ID'].tolist()]
history_column["rating"] = df_data['history rating'].tolist()
history_column["hist length"] = [len(x) for x in history_column["rating"]]

user_seq = {
    "history ID": {
        "train": history_column["ID"][:train_num],
        "valid": history_column["ID"][train_num:train_num + valid_num],
        "test": history_column["ID"][train_num + valid_num:],
    },
    "history rating": {
        "train": history_column["rating"][:train_num],
        "valid": history_column["rating"][train_num:train_num + valid_num],
        "test": history_column["rating"][train_num + valid_num:],
    },
    "history length": {
        "train": history_column["hist length"][:train_num],
        "valid": history_column["hist length"][train_num:train_num + valid_num],
        "test": history_column["hist length"][train_num + valid_num:],
    },
}


# In[ ]:


# field_rec = ["User ID", "Movie ID", "rating"]
# df_rec = df_data[field_rec]
# df_train_rec = df_rec[:train_num].sort_values(by=["User ID", "Movie ID"], inplace=False, kind="stable")
# df_test_rec = df_rec[train_num + valid_num:].sort_values(by=["User ID", "Movie ID"], inplace=False, kind="stable")
# df_train_rec.to_csv(os.path.join(target_dir, "train.txt"), sep=' ', index=False, header=None)
# df_test_rec.to_csv(os.path.join(target_dir, "test.txt"), sep=' ', index=False, header=None)


# In[162]:


# Save train/valid/test in parquet format

df_train = df_data[:train_num].reset_index(drop=True)
df_valid = df_data[train_num:train_num + valid_num].reset_index(drop=True)
df_test = df_data[train_num + valid_num:].reset_index(drop=True)

assert len(df_train) == train_num
assert len(df_valid) == valid_num
assert len(df_test) == test_num

print(f"Train num: {len(df_train)}")
print(f"Valid num: {len(df_valid)}")
print(f"Test num: {len(df_test)}")

df_train.to_parquet(os.path.join(target_dir, "train.parquet.gz"), compression="gzip")
df_valid.to_parquet(os.path.join(target_dir, "valid.parquet.gz"), compression="gzip")
df_test.to_parquet(os.path.join(target_dir, "test.parquet.gz"), compression="gzip")


# In[163]:


# Re-read for sanity check

train_dataset = pd.read_parquet(os.path.join(target_dir, "train.parquet.gz"))
valid_dataset = pd.read_parquet(os.path.join(target_dir, "valid.parquet.gz"))
test_dataset = pd.read_parquet(os.path.join(target_dir, "test.parquet.gz"))


# In[164]:


# Convert df_data to CTR data via feature_dict

ctr_X, ctr_Y = [], []
for idx, row in tqdm(df_data.iterrows()):
    ctr_X.append([feature_dict[field][row[field]] for field in field_names])
    ctr_Y.append(int(row["labels"]))

ctr_X = np.array(ctr_X)
ctr_Y = np.array(ctr_Y)
print("ctr_X", ctr_X.shape)
print("ctr_Y", ctr_Y.shape)
feature_count_np = np.array(feature_count).reshape(1, -1)
assert (ctr_X - feature_count_np <= 0).sum() == ctr_X.shape[0] * ctr_X.shape[1]
assert (ctr_Y == 0).sum() + (ctr_Y == 1).sum() == ctr_Y.shape[0]


# In[165]:


# Truncate the user sequence up to 30, i.e., 5 <= length <= 30.

import torch
from torch.nn.utils.rnn import pad_sequence

user_seq_trunc = {
    "history ID": {}, 
    "history rating": {}, 
    "history mask": {}, 
}

for hist_name in user_seq:
    for split in user_seq[hist_name]:
        if hist_name != "history length":
            user_seq_trunc[hist_name][split] = pad_sequence(
                [torch.tensor(x[-30:]) for x in user_seq[hist_name][split]], 
                batch_first=True, 
            )
        else:
            user_seq_trunc["history mask"][split] = pad_sequence(
                [torch.ones(min(x, 30)) for x in user_seq[hist_name][split]], 
                batch_first=True, 
            )


# In[166]:


# Save CTR data & truncated user sequence into one .h5 file

with h5py.File(os.path.join(target_dir, f"ctr.h5"), "w") as hf:
    hf.create_dataset("train data", data=ctr_X[:train_num, :])
    hf.create_dataset("valid data", data=ctr_X[train_num:train_num + valid_num, :])
    hf.create_dataset("test data", data=ctr_X[train_num + valid_num:, :])
    hf.create_dataset("train label", data=ctr_Y[:train_num])
    hf.create_dataset("valid label", data=ctr_Y[train_num:train_num + valid_num])
    hf.create_dataset("test label", data=ctr_Y[train_num + valid_num:])
    for hist_name in user_seq_trunc:
        for split in user_seq_trunc[hist_name]:
            hf.create_dataset(f"{split} {hist_name}", data=user_seq_trunc[hist_name][split])

with h5py.File(os.path.join(target_dir, f"ctr.h5"), "r") as hf:
    assert (ctr_X - np.concatenate([hf["train data"][:], hf["valid data"][:], hf["test data"][:]], axis=0)).sum() == 0
    assert (ctr_Y - np.concatenate([hf["train label"][:], hf["valid label"][:], hf["test label"][:]], axis=0)).sum() == 0
    for hist_name in user_seq_trunc:
        for split in user_seq_trunc[hist_name]:
            assert (user_seq_trunc[hist_name][split] - hf[f"{split} {hist_name}"][:]).sum() == 0

    x = hf["train data"][:]
    assert (x - ctr_X[:train_num, :]).sum() == 0
    print(f"train data: {x.shape}")
    
    x = hf["valid data"][:]
    assert (x - ctr_X[train_num:train_num + valid_num, :]).sum() == 0
    print(f"valid data: {x.shape}")
    
    x = hf["test data"][:]
    assert (x - ctr_X[train_num + valid_num:, :]).sum() == 0
    print(f"test data: {x.shape}")
    
    x = hf["train label"][:]
    assert (x - ctr_Y[:train_num]).sum() == 0
    print(f"train label: {x.shape}")
    
    x = hf["valid label"][:]
    assert (x - ctr_Y[train_num:train_num + valid_num]).sum() == 0
    print(f"valid label: {x.shape}")
    
    x = hf["test label"][:]
    assert (x - ctr_Y[train_num + valid_num:]).sum() == 0
    print(f"test label: {x.shape}")


# In[167]:


movie_counts = df_data['Movie ID'].value_counts()


# In[168]:


# train_set['Movie ID'].value_counts()
multi_occurrences = movie_counts[movie_counts > 1].index
print(multi_occurrences)


# In[170]:


# Remove the timestamps
field_rec = ["User ID", "Movie ID", "rating"]
df_ratings = df_data[field_rec]

# Group by each user and filter for users with more than 10 ratings
user_groups = df_ratings.groupby('User ID').filter(lambda x: len(x) > 5)

# Extract the last record for each user in the test set and the rest as training
tail_set = user_groups.groupby('User ID').tail(1)
test_set = tail_set[tail_set['Movie ID'].isin(multi_occurrences)]
train_set = df_ratings.drop(test_set.index)

# Display the shapes to confirm the split
print("Training set shape:", train_set.shape)
print("Test set shape:", test_set.shape)


# In[16]:





# In[171]:


# Save the data as .txt files (1-based integer encoding for LightGCN)
df_train_rec = train_set.sort_values(by=["User ID", "Movie ID"], inplace=False, kind="stable")
df_test_rec = test_set.sort_values(by=["User ID", "Movie ID"], inplace=False, kind="stable")

encoded_train_users = [feature_dict["User ID"][u] + 1 for u in df_train_rec["User ID"]]
encoded_train_items = [feature_dict["Movie ID"][m] + 1 for m in df_train_rec["Movie ID"]]
encoded_test_users = [feature_dict["User ID"][u] + 1 for u in df_test_rec["User ID"]]
encoded_test_items = [feature_dict["Movie ID"][m] + 1 for m in df_test_rec["Movie ID"]]

with open(os.path.join(target_dir, "train.txt"), "w") as f:
    for u, i, r in zip(encoded_train_users, encoded_train_items, df_train_rec["rating"]):
        f.write(f"{u} {i} {r}\n")

with open(os.path.join(target_dir, "test.txt"), "w") as f:
    for u, i, r in zip(encoded_test_users, encoded_test_items, df_test_rec["rating"]):
        f.write(f"{u} {i} {r}\n")

