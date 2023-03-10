import argparse
import os
import csv
from pathlib import Path

def create_dataset(args):
    dataset_directory = os.path.join("re-net", "data", "ICEWS14")
    full_dataset_path = os.path.join(dataset_directory, "resources", "full.txt")
    train_dataset_path = os.path.join(dataset_directory, "train.txt")
    valid_dataset_path = os.path.join(dataset_directory, "valid.txt")
    test_dataset_path = os.path.join(dataset_directory, "test.txt")
    train_names_dataset_path = os.path.join(dataset_directory, "resources", "train_names.txt")
    valid_names_dataset_path = os.path.join(dataset_directory, "resources", "valid_names.txt")
    test_names_dataset_path = os.path.join(dataset_directory, "resources", "test_names.txt")
    entity2id_path = os.path.join(dataset_directory, "entity2id.txt")
    relation2id_path = os.path.join(dataset_directory, "relation2id.txt")
    timestamp2id_path = os.path.join(dataset_directory, "timestamp2id.txt")
    
    create_2id_files(full_dataset_path, entity2id_path, relation2id_path, timestamp2id_path)
    create_train_valid_test(entity2id_path, relation2id_path, timestamp2id_path, 
                            train_names_dataset_path, valid_names_dataset_path, test_names_dataset_path, 
                            train_dataset_path, valid_dataset_path, test_dataset_path)
    
def create_2id_files(full_dataset_path, entity2id_path, relation2id_path, timestamp2id_path):
    rows = []
    entity2id = []
    relation2id = []
    timestamp2id = ["-"]
    with open(full_dataset_path, encoding='utf-8') as full_dataset:
        records = csv.DictReader(full_dataset, delimiter='\t')
        for row in records:
            rows.append(row)
    
    for row in rows:
        if row["head"] not in entity2id:
            entity2id.append(row["head"])
        if row["relation"] not in relation2id:
            relation2id.append(row["relation"])
        if row["tail"] not in entity2id:
            entity2id.append(row["tail"])
        if row["timestamp"] not in timestamp2id:
            timestamp2id.append(row["timestamp"])
    
    entity2id.sort()
    relation2id.sort()
    timestamp2id.sort()

    entity2id_str = ""
    for i in range(len(entity2id)):
        entity2id_str += entity2id[i] + "\t" + str(i) + "\n"
    relation2id_str = ""
    for i in range(len(relation2id)):
        relation2id_str += relation2id[i] + "\t" + str(i) + "\n"
    timestamp2id_str = ""
    for i in range(len(timestamp2id)):
        timestamp2id_str += timestamp2id[i] + "\t" + str(i) + "\n"

    write(entity2id_path, entity2id_str)
    write(relation2id_path, relation2id_str)
    write(timestamp2id_path, timestamp2id_str)

def create_train_valid_test(entity2id_path, relation2id_path, timestamp2id_path, 
                            train_names_dataset_path, valid_names_dataset_path, test_names_dataset_path, 
                            train_dataset_path, valid_dataset_path, test_dataset_path):
    entity2id = {}
    relation2id = {}
    timestamp2id = {}

    with open(entity2id_path, encoding='utf-8') as dataset:
        records = csv.DictReader(dataset, fieldnames=["entity", "id"], delimiter='\t')
        for row in records:
            entity2id[row["entity"]] = int(row["id"])
    with open(relation2id_path, encoding='utf-8') as dataset:
        records = csv.DictReader(dataset, fieldnames=["relation", "id"], delimiter='\t')
        for row in records:
            relation2id[row["relation"]] = int(row["id"])
    with open(timestamp2id_path, encoding='utf-8') as dataset:
        records = csv.DictReader(dataset, fieldnames=["timestamp", "id"], delimiter='\t')
        for row in records:
            timestamp2id[row["timestamp"]] = int(row["id"])
    
    test_str = ""
    train_str = ""
    valid_str = ""

    test_rows = []
    with open(test_names_dataset_path, encoding='utf-8') as dataset:
        records = csv.DictReader(dataset, fieldnames=["head", "relation", "tail", "timestamp"], delimiter='\t')
        for row in records:
            test_rows.append(
                {
                    "head": name2id(row["head"], entity2id),
                    "relation": name2id(row["relation"], relation2id),
                    "tail": name2id(row["tail"], entity2id),
                    "time_from": name2id(row["timestamp"], timestamp2id),
                    "time_to": name2id("-", timestamp2id)
                })
    test_rows.sort(key=lambda row: (row["time_from"], row["head"], row["relation"], row["tail"], row["time_to"]))
    for row in test_rows:
        test_str += str(row["head"]) + "\t" \
            + str(row["relation"]) + "\t" \
            + str(row["tail"]) + "\t" \
            + str(row["time_from"]) + "\t" \
            + str(row["time_to"]) + "\n"
        
        
    train_rows = []
    with open(train_names_dataset_path, encoding='utf-8') as dataset:
        records = csv.DictReader(dataset, fieldnames=["head", "relation", "tail", "timestamp"], delimiter='\t')
        for row in records:
            train_rows.append(
                {
                    "head": name2id(row["head"], entity2id),
                    "relation": name2id(row["relation"], relation2id),
                    "tail": name2id(row["tail"], entity2id),
                    "time_from": name2id(row["timestamp"], timestamp2id),
                    "time_to": name2id("-", timestamp2id)
                })
    train_rows.sort(key=lambda row: (row["time_from"], row["head"], row["relation"], row["tail"], row["time_to"]))
    for row in train_rows:
        train_str += str(row["head"]) + "\t" \
            + str(row["relation"]) + "\t" \
            + str(row["tail"]) + "\t" \
            + str(row["time_from"]) + "\t" \
            + str(row["time_to"]) + "\n"

    
    valid_rows = []
    with open(valid_names_dataset_path, encoding='utf-8') as dataset:
        records = csv.DictReader(dataset, fieldnames=["head", "relation", "tail", "timestamp"], delimiter='\t')
        for row in records:
            valid_rows.append(
                {
                    "head": name2id(row["head"], entity2id),
                    "relation": name2id(row["relation"], relation2id),
                    "tail": name2id(row["tail"], entity2id),
                    "time_from": name2id(row["timestamp"], timestamp2id),
                    "time_to": name2id("-", timestamp2id)
                })
    valid_rows.sort(key=lambda row: (row["time_from"], row["head"], row["relation"], row["tail"], row["time_to"]))
    for row in valid_rows:
        valid_str += str(row["head"]) + "\t" \
            + str(row["relation"]) + "\t" \
            + str(row["tail"]) + "\t" \
            + str(row["time_from"]) + "\t" \
            + str(row["time_to"]) + "\n"
    
    write(test_dataset_path, test_str)
    write(train_dataset_path, train_str)
    write(valid_dataset_path, valid_str)

def name2id(name, dir):
    return dir[name]

def touch(path):
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.makedirs(dir)
    Path(path).touch(exist_ok=True)

def write(path, text):
    touch(path)
    out_file = open(path, "w", encoding="utf8")
    out_file.write(text)
    out_file.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RENet')

    args = parser.parse_args()
    print(args)
    create_dataset(args)

