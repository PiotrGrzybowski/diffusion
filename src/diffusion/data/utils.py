from torch.utils.data import Dataset, Subset


def create_filtered_dataset(dataset: Dataset, labels: list[int], samples: int):
    indices = []
    label_counts = {label: 0 for label in labels}

    for idx, (_, label) in enumerate(dataset):
        if label in labels and label_counts[label] < samples:
            indices.append(idx)
            label_counts[label] += 1

            if all(count >= samples for count in label_counts.values()):
                break

    subset = Subset(dataset, indices)

    return subset
