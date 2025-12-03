from huggingface_hub import list_datasets


if __name__ == "__main__":
    with open("datasets_list.txt", "w") as f:
        for dataset in list_datasets():
            f.write(f"{dataset.id}\n")
