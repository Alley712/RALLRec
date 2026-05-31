#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import json
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

dataset_name = "ml-1m"
root = f"../data/{dataset_name}"
source_dir = os.path.join(root, "raw_data")
target_dir = os.path.join(root, "proc_data")

os.makedirs(target_dir, exist_ok=True)


# In[2]:


age_dict = {
    1: "under 18",
    18: "18-24",
    25: "25-34",
    35: "35-44",
    45: "45-49",
    50: "50-55",
    56: "above 56"
}

job_dict = {
    0: "other or not specified",
	1: "academic/educator",
	2: "artist",
	3: "clerical/admin",
	4: "college/grad student",
	5: "customer service",
	6: "doctor/health care",
	7: "executive/managerial",
	8: "farmer",
	9: "homemaker",
	10: "K-12 student",
	11: "lawyer",
	12: "programmer",
	13: "retired",
	14: "sales/marketing",
	15: "scientist",
	16: "self-employed",
	17: "technician/engineer",
	18: "tradesman/craftsman",
	19: "unemployed",
	20: "writer",
}


# In[3]:


# User data

user_data = []
user_fields = ["User ID", "Gender", "Age", "Job", "Zipcode"]
for line in open(os.path.join(source_dir, "users.dat"), "r").readlines():
    ele = line.strip().split("::")
    user_id, gender, age, job, zipcode = [x.strip() for x in ele]
    gender = "male" if gender == "M" else "female"
    age = age_dict[int(age)]
    job = job_dict[int(job)]
    user_data.append([int(user_id), gender, age, job, zipcode])

df_user = pd.DataFrame(user_data, columns=user_fields)
print(f"Total number of users: {len(df_user)}")


# In[4]:


# Movie data

movie_data = []
movie_detail = {}
movie_fields = ["Movie ID", "Movie title", "Movie genre"]
for line in open(os.path.join(source_dir, "movies.dat"), "r", encoding="ISO-8859-1").readlines():
    ele = line.strip().split("::")
    movie_id = int(ele[0].strip())
    movie_title = ele[1].strip()
    movie_genre = ele[2].strip().split("|")[0]
    movie_data.append([movie_id, movie_title, movie_genre])
    movie_detail[movie_id] = [movie_title, movie_genre]

df_movie = pd.DataFrame(movie_data, columns=movie_fields)
print(f"Total number of movies: {len(df_movie)}")

json.dump(movie_detail, open(os.path.join(target_dir, "movie_detail.json"), "w"))


# In[5]:


# Rating data

rating_data = []
rating_fields = ["User ID", "Movie ID", "rating", "timestamp", "labels"]
user_list, movie_list = list(df_user["User ID"]), list(df_movie["Movie ID"])
for line in open(os.path.join(source_dir, "ratings.dat"), "r").readlines():
    ele = [x.strip() for x in line.strip().split("::")] 
    user, movie, rating, timestamp = int(ele[0]), int(ele[1]), int(ele[2]), int(ele[3])
    label = 1 if rating > 3 else 0
    if user in user_list and movie in movie_list:
        rating_data.append([user, movie, rating, timestamp, label])

df_ratings = pd.DataFrame(rating_data, columns=rating_fields)
print(f"Total number of ratings: {len(df_ratings)}")


# In[7]:


# Merge df_user/df_movie/df_rating into df_data

df_data = pd.merge(df_ratings, df_user, on=["User ID"], how="inner")
df_data = pd.merge(df_data, df_movie, on=["Movie ID"], how="inner")

df_data.sort_values(by=["timestamp", "User ID", "Movie ID"], inplace=True, kind="stable")

field_names = ["timestamp", "User ID", "Gender", "Age", "Job", "Zipcode", "Movie ID", "Movie title", "Movie genre", "rating", "labels"]

df_data = df_data[field_names].reset_index(drop=True)

df_data.head()


# In[9]:


# Encode the feature dict for CTR data

def add_to_dict(dict, feature):
    if feature not in dict:
        dict[feature] = len(dict)

field_names = ["User ID", "Gender", "Age", "Job", "Zipcode", "Movie ID", "Movie title", "Movie genre"]
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


# In[9]:


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


# In[10]:


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


# In[11]:


# Split & save user history sequence

train_num = int(0.8 * len(df_data))
valid_num = int(0.1 * len(df_data))
test_num = len(df_data) - train_num - valid_num

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


# In[32]:


field_rec = ["User ID", "Movie ID", "rating"]
df_rec = df_data[field_rec]
df_train_rec = df_rec[:train_num].sort_values(by=["User ID", "Movie ID"], inplace=False, kind="stable")
df_test_rec = df_rec[train_num + valid_num:].sort_values(by=["User ID", "Movie ID"], inplace=False, kind="stable")
df_train_rec.to_csv(os.path.join(target_dir, "train.txt"), sep=' ', index=False, header=None)
df_test_rec.to_csv(os.path.join(target_dir, "test.txt"), sep=' ', index=False, header=None)


# In[13]:


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


# In[14]:


# Re-read for sanity check

train_dataset = pd.read_parquet(os.path.join(target_dir, "train.parquet.gz"))
valid_dataset = pd.read_parquet(os.path.join(target_dir, "valid.parquet.gz"))
test_dataset = pd.read_parquet(os.path.join(target_dir, "test.parquet.gz"))


# In[15]:


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


# In[16]:


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


# In[17]:


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


# In[18]:


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


# In[2]:


# Load the data
# Assuming the dataset has columns 'user_id', 'movie_id', 'rating', and 'timestamp'
# Replace 'path_to_dataset' with the actual path to your dataset file
rating_data = []
rating_fields = ["User ID", "Movie ID", "rating", "timestamp"]
for line in open(os.path.join(source_dir, "ratings.dat"), "r").readlines():
    ele = [x.strip() for x in line.strip().split("::")] 
    user, movie, rating, timestamp = int(ele[0]), int(ele[1]), int(ele[2]), int(ele[3])
    rating_data.append([user, movie, rating, timestamp])

df_ratings = pd.DataFrame(rating_data, columns=rating_fields)
print("Data set shape:", df_ratings.shape)

# Sort by user_id and timestamp
df_ratings = df_ratings.sort_values(by=['User ID', 'timestamp'])


# In[5]:


movie_counts = df_ratings['Movie ID'].value_counts()


# In[13]:


# train_set['Movie ID'].value_counts()
multi_occurrences = movie_counts[movie_counts > 1].index
print(multi_occurrences)


# In[18]:


# Remove the timestamps
field_rec = ["User ID", "Movie ID", "rating"]
df_ratings = df_ratings[field_rec]

# Group by each user and filter for users with more than 10 ratings
user_groups = df_ratings.groupby('User ID').filter(lambda x: len(x) > 10)

# Extract the last record for each user in the test set and the rest as training
tail_set = user_groups.groupby('User ID').tail(1)
test_set = tail_set[tail_set['Movie ID'].isin(multi_occurrences)]
train_set = df_ratings.drop(test_set.index)

# Display the shapes to confirm the split
print("Training set shape:", train_set.shape)
print("Test set shape:", test_set.shape)


# In[16]:





# In[19]:


# Save the data as .txt files
df_train_rec = train_set.sort_values(by=["User ID", "Movie ID"], inplace=False, kind="stable")
df_test_rec = test_set.sort_values(by=["User ID", "Movie ID"], inplace=False, kind="stable")
df_train_rec.to_csv(os.path.join(target_dir, "train.txt"), sep=' ', index=False, header=None)
df_test_rec.to_csv(os.path.join(target_dir, "test.txt"), sep=' ', index=False, header=None)

