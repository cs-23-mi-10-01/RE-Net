import argparse
import os
import csv
from pathlib import Path

def create_dataset(args):
    dataset_directory = os.path.join("re-net", "data", "YAGO")
    full_named_dataset_path = os.path.join(dataset_directory, "resources", "full_named.txt")
    train_named_dataset_path = os.path.join(dataset_directory, "resources", "train_named.txt")
    valid_named_dataset_path = os.path.join(dataset_directory, "resources", "valid_named.txt")
    test_named_dataset_path = os.path.join(dataset_directory, "resources", "test_named.txt")
    test_resource_dataset_path = os.path.join(dataset_directory, "resources", "test.txt")
    train_resource_dataset_path = os.path.join(dataset_directory, "resources", "train.txt")
    valid_resource_dataset_path = os.path.join(dataset_directory, "resources", "valid.txt")
    full_named_timestamps_dataset_path = os.path.join(dataset_directory, "resources", "full_named_timestamps.txt")
    relation2id_resources_path = os.path.join(dataset_directory, "resources", "relation2id.txt")
    entity2id_resources_path = os.path.join(dataset_directory, "resources", "entity2id.txt")
    train_dataset_path = os.path.join(dataset_directory, "train.txt")
    valid_dataset_path = os.path.join(dataset_directory, "valid.txt")
    test_dataset_path = os.path.join(dataset_directory, "test.txt")
    stat_path = os.path.join(dataset_directory, "stat.txt")
    entity2id_path = os.path.join(dataset_directory, "entity2id.txt")
    relation2id_path = os.path.join(dataset_directory, "relation2id.txt")
    timestamp2id_path = os.path.join(dataset_directory, "timestamp2id.txt")
    
    create_full_named(full_named_timestamps_dataset_path, entity2id_resources_path, 
                      relation2id_resources_path, full_named_dataset_path)
    create_2id_files(full_named_dataset_path, entity2id_path, relation2id_path, timestamp2id_path, stat_path)
    create_named_train_test_valid(train_named_dataset_path, test_named_dataset_path, valid_named_dataset_path,
                                  train_resource_dataset_path, test_resource_dataset_path, valid_resource_dataset_path,
                                  entity2id_resources_path, relation2id_resources_path)
    create_train_valid_test(entity2id_path, relation2id_path, timestamp2id_path, 
                            train_named_dataset_path, valid_named_dataset_path, test_named_dataset_path, 
                            train_dataset_path, valid_dataset_path, test_dataset_path)
    
def create_full_named(full_named_timestamps_dataset_path, entity2id_resources_path, 
                      relation2id_resources_path, full_named_dataset_path):
    
    full_named_timestamps_rows = []
    with open(full_named_timestamps_dataset_path, encoding='utf-8') as full_dataset:
        records = csv.DictReader(full_dataset, fieldnames=["head", "relation", "tail", "time_from", "time_to"], delimiter='\t')
        for row in records:
            full_named_timestamps_rows.append(row)
    
    id2entity = {}
    id2relation = {}

    with open(entity2id_resources_path, encoding='utf-8') as dataset:
        records = csv.DictReader(dataset, fieldnames=["entity", "id"], delimiter='\t')
        for row in records:
            id2entity[row["id"]] = row["entity"]
    with open(relation2id_resources_path, encoding='utf-8') as dataset:
        records = csv.DictReader(dataset, fieldnames=["relation", "id"], delimiter='\t')
        for row in records:
            id2relation[row["id"]] = row["relation"]

    full_named_rows_str = ""
    for row in full_named_timestamps_rows:
        full_named_rows_str += id2name(row["head"], id2entity) + "\t" + \
            id2name(row["relation"], id2relation) + "\t" + \
            id2name(row["tail"], id2entity) + "\t" + \
            date_to_year_only(row["time_from"]) + "\t" + \
            date_to_year_only(row["time_to"]) + "\n"
        
    write(full_named_dataset_path, full_named_rows_str)
    

def date_to_year_only(date):
    if date[3] == '-':
        return "0" + date[0] + date[1] + date[2] + "-##-##"
    return date[0] + date[1] + date[2] + date[3] + "-##-##"

def create_2id_files(full_named_dataset_path, entity2id_path, relation2id_path, timestamp2id_path, stat_path):
    rows = []
    entity2id = []
    relation2id = []
    timestamp2id = []
    with open(full_named_dataset_path, encoding='utf-8') as full_dataset:
        records = csv.DictReader(full_dataset, fieldnames=["head", "relation", "tail", "time_from", "time_to"], delimiter='\t')
        for row in records:
            rows.append(row)
    
    for row in rows:
        if row["head"] not in entity2id:
            entity2id.append(row["head"])
        if row["relation"] not in relation2id:
            relation2id.append(row["relation"])
        if row["tail"] not in entity2id:
            entity2id.append(row["tail"])
        if row["time_from"] not in timestamp2id:
            timestamp2id.append(row["time_from"])
        if row["time_to"] not in timestamp2id:
            timestamp2id.append(row["time_to"])
    
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

    stat_str = str(len(entity2id)) + "\t" + str(len(relation2id)) + "\t" + str(len(timestamp2id)) + "\n"
    
    write(stat_path, stat_str)

def create_named_train_test_valid(train_named_dataset_path, test_named_dataset_path, valid_named_dataset_path,
                                  train_resource_dataset_path, test_resource_dataset_path, valid_resource_dataset_path,
                                  entity2id_resources_path, relation2id_resources_path):    
    id2entity = {}
    id2relation = {}

    with open(entity2id_resources_path, encoding='utf-8') as dataset:
        records = csv.DictReader(dataset, fieldnames=["entity", "id"], delimiter='\t')
        for row in records:
            id2entity[row["id"]] = row["entity"]
    with open(relation2id_resources_path, encoding='utf-8') as dataset:
        records = csv.DictReader(dataset, fieldnames=["relation", "id"], delimiter='\t')
        for row in records:
            id2relation[row["id"]] = row["relation"]

    train_resource_rows = []
    with open(train_resource_dataset_path, encoding='utf-8') as full_dataset:
        records = csv.DictReader(full_dataset, fieldnames=["head", "relation", "tail", "time_from", "time_to"], delimiter='\t')
        for row in records:
            train_resource_rows.append({
                "head": row["head"],
                "relation": row["relation"],
                "tail": row["tail"],
                "time_from": date_to_year_only(row["time_from"]),
                "time_to": date_to_year_only(row["time_to"])
            })

    train_resource_rows.sort(key=lambda row: (row["time_from"], row["head"], row["relation"], row["tail"], row["time_to"]))
    train_named_rows_str = ""
    for row in train_resource_rows:
        train_named_rows_str += id2name(row["head"], id2entity) + "\t" + \
            id2name(row["relation"], id2relation) + "\t" + \
            id2name(row["tail"], id2entity) + "\t" + \
            date_to_year_only(row["time_from"]) + "\t" + \
            date_to_year_only(row["time_to"]) + "\n"

    test_resource_rows = []
    with open(test_resource_dataset_path, encoding='utf-8') as full_dataset:
        records = csv.DictReader(full_dataset, fieldnames=["head", "relation", "tail", "time_from", "time_to"], delimiter='\t')
        for row in records:
            test_resource_rows.append({
                "head": row["head"],
                "relation": row["relation"],
                "tail": row["tail"],
                "time_from": date_to_year_only(row["time_from"]),
                "time_to": date_to_year_only(row["time_to"])
            })

    test_resource_rows.sort(key=lambda row: (row["time_from"], row["head"], row["relation"], row["tail"], row["time_to"]))
    test_named_rows_str = ""
    for row in test_resource_rows:
        test_named_rows_str += id2name(row["head"], id2entity) + "\t" + \
            id2name(row["relation"], id2relation) + "\t" + \
            id2name(row["tail"], id2entity) + "\t" + \
            date_to_year_only(row["time_from"]) + "\t" + \
            date_to_year_only(row["time_to"]) + "\n"

    valid_resource_rows = []
    with open(valid_resource_dataset_path, encoding='utf-8') as full_dataset:
        records = csv.DictReader(full_dataset, fieldnames=["head", "relation", "tail", "time_from", "time_to"], delimiter='\t')
        for row in records:
            valid_resource_rows.append({
                "head": row["head"],
                "relation": row["relation"],
                "tail": row["tail"],
                "time_from": date_to_year_only(row["time_from"]),
                "time_to": date_to_year_only(row["time_to"])
            })

    valid_resource_rows.sort(key=lambda row: (row["time_from"], row["head"], row["relation"], row["tail"], row["time_to"]))
    valid_named_rows_str = ""
    for row in valid_resource_rows:
        valid_named_rows_str += id2name(row["head"], id2entity) + "\t" + \
            id2name(row["relation"], id2relation) + "\t" + \
            id2name(row["tail"], id2entity) + "\t" + \
            date_to_year_only(row["time_from"]) + "\t" + \
            date_to_year_only(row["time_to"]) + "\n"
        
    write(train_named_dataset_path, train_named_rows_str)
    write(test_named_dataset_path, test_named_rows_str)
    write(valid_named_dataset_path, valid_named_rows_str)

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
        records = csv.DictReader(dataset, fieldnames=["head", "relation", "tail", "time_from", "time_to"], delimiter='\t')
        for row in records:
            test_rows.append(
                {
                    "head": name2id(row["head"], entity2id),
                    "relation": name2id(row["relation"], relation2id),
                    "tail": name2id(row["tail"], entity2id),
                    "time_from": name2id(row["time_from"], timestamp2id),
                    "time_to": name2id(row["time_to"], timestamp2id)
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
        records = csv.DictReader(dataset, fieldnames=["head", "relation", "tail", "time_from", "time_to"], delimiter='\t')
        for row in records:
            train_rows.append(
                {
                    "head": name2id(row["head"], entity2id),
                    "relation": name2id(row["relation"], relation2id),
                    "tail": name2id(row["tail"], entity2id),
                    "time_from": name2id(row["time_from"], timestamp2id),
                    "time_to": name2id(row["time_to"], timestamp2id)
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
        records = csv.DictReader(dataset, fieldnames=["head", "relation", "tail", "time_from", "time_to"], delimiter='\t')
        for row in records:
            valid_rows.append(
                {
                    "head": name2id(row["head"], entity2id),
                    "relation": name2id(row["relation"], relation2id),
                    "tail": name2id(row["tail"], entity2id),
                    "time_from": name2id(row["time_from"], timestamp2id),
                    "time_to": name2id(row["time_to"], timestamp2id)
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

def id2name(name, dir):
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

