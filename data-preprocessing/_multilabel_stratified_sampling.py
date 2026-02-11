# from https://www.nature.com/articles/s41597-020-0495-6#Sec20
# modified to actually implement the quality control described in the docstring

import numpy as np


def stratify(
    data, classes, ratios, qualities, ecgs_per_patient, nr_clean_folds=1, random_seed=0
):
    """Stratifying procedure. Modified from https://vict0rs.ch/2018/05/24/sample-multilabel-dataset/ (based on Sechidis 2011)

    data is a list of lists: a list of labels, for each sample.
    Each sample's labels should be ints, if they are one-hot
    encoded, use one_hot = True

    classes is the list of classes each label can take

    ratios is a list, summing to 1, of how the dataset should be split

    qualities: quality per entry (only >0 can be assigned to clean
    folds; 4 will always be assigned to final fold)

    ecgs_per_patient: list with number of ecgs per sample

    nr_clean_folds: the last nr_clean_folds can only take clean entries
    """
    np.random.seed(random_seed)  # fix the random seed

    # data is now always a list of lists; len(data) is the number
    # of patients; data[i] is the list of all labels for patient i
    # (possibly multiple identical entries)
    # size is the number of ecgs
    size = np.sum(ecgs_per_patient)

    # Organize data per label: for each label l, per_label_data[l]
    # contains the list of patients in data which have this label
    # (potentially multiple identical entries)
    per_label_data = {c: [] for c in classes}
    for i, d in enumerate(data):
        for l in d:
            per_label_data[l].append(i)

    # In order not to compute lengths each time, they are tracked here.
    subset_sizes = [r * size for r in ratios]  # list of subset_sizes in terms of ecgs
    per_label_subset_sizes = {
        c: [r * len(per_label_data[c]) for r in ratios] for c in classes
    }  # dictionary with label: list of subset sizes in terms of patients

    # For each subset we want, the set of sample-ids which should end up in it
    stratified_data_ids = [set() for _ in range(len(ratios))]  # initialize empty

    # For each sample in the data set
    print("Assigning patients to folds...")
    size_prev = size + 1  # just for output
    while size > 0:
        if int(size_prev / 1000) > int(size / 1000):
            print(
                "Remaining patients/ecgs to distribute:",
                size,
                "non-empty labels:",
                np.sum(
                    [
                        1
                        for l, label_data in per_label_data.items()
                        if len(label_data) > 0
                    ]
                ),
            )
            size_prev = size

        # Compute |Di|
        lengths = {
            l: len(label_data) for l, label_data in per_label_data.items()
        }  # dictionary label: number of ecgs with this label that have not been assigned to a fold yet

        try:
            # Find label of smallest |Di|
            label = min({k: v for k, v in lengths.items() if v > 0}, key=lengths.get)
        except ValueError:
            # If the dictionary in 'min' is empty we get a ValueError.
            # This can happen if there are unlabeled samples.
            # In this case, 'size' would be > 0 but only samples without label would remain.
            # "No label" could be a class in itself: it's up to you to format your data accordingly.
            break

        # For each patient with label 'label' get patient and corresponding counts
        unique_samples, unique_counts = np.unique(
            per_label_data[label], return_counts=True
        )
        idxs_sorted = np.argsort(unique_counts, kind="stable")[::-1]
        unique_samples = unique_samples[
            idxs_sorted
        ]  # this is a list of all patient ids with this label sorted by size descending
        unique_counts = unique_counts[idxs_sorted]  # these are the corresponding counts

        # loop through all patient ids with this label
        for current_id, current_count in zip(unique_samples, unique_counts):
            subset_sizes_for_label = per_label_subset_sizes[
                label
            ]  # current subset sizes for the chosen label

            # if quality is bad remove clean folds (i.e. sample cannot be assigned to clean folds)
            if qualities[current_id] < 1:
                subset_sizes_for_label = subset_sizes_for_label[
                    : len(ratios) - nr_clean_folds
                ]
            elif qualities[current_id] > 3:
                # Set all folds except the last to 0 (no capacity)
                subset_sizes_for_label = [0] * (len(ratios) - 1) + [
                    subset_sizes_for_label[-1]
                ]

            # Find argmax clj i.e. subset in greatest need of the current label
            largest_subsets = np.argwhere(
                subset_sizes_for_label == np.amax(subset_sizes_for_label)
            ).flatten()

            # if there is a single best choice: assign it
            if len(largest_subsets) == 1:
                subset = largest_subsets[0]
            # If there is more than one such subset, find the one in greatest need of any label
            else:
                largest_subsets2 = np.argwhere(
                    np.array(subset_sizes)[largest_subsets]
                    == np.amax(np.array(subset_sizes)[largest_subsets])
                ).flatten()
                subset = largest_subsets[np.random.choice(largest_subsets2)]

            # Store the sample's id in the selected subset
            stratified_data_ids[subset].add(current_id)

            # There is current_count fewer samples to distribute
            size -= ecgs_per_patient[current_id]

            # The selected subset needs current_count fewer samples
            subset_sizes[subset] -= ecgs_per_patient[current_id]

            # In the selected subset, there is one more example for each label the current sample has
            for l in data[current_id]:
                per_label_subset_sizes[l][subset] -= 1

            # Remove the sample from the dataset, meaning from all per_label dataset created
            for x in per_label_data.keys():
                per_label_data[x] = [y for y in per_label_data[x] if y != current_id]

    # Create the stratified dataset as a list of subsets, each containing the original labels
    stratified_data_ids = [sorted(strat) for strat in stratified_data_ids]
    stratified_data = [[data[i] for i in strat] for strat in stratified_data_ids]

    # Return both the stratified indexes, to be used to sample the 'features' associated with your labels
    # And the stratified labels dataset
    return stratified_data_ids, stratified_data


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Example: Stratify a medical dataset with multiple ECG recordings per patient

    # Patient data: each patient has a list of labels (diagnoses)
    # Patient 0 has labels [0, 1] (e.g., atrial fibrillation and tachycardia)
    # Patient 1 has label [2] (e.g., normal sinus rhythm)
    # Patient 2 has labels [0, 2] (e.g., atrial fibrillation and normal sinus rhythm)
    # Patient 3 has label [1] (e.g., tachycardia)
    # Patient 4 has labels [0, 1, 2] (multiple diagnoses)
    data = [
        [0, 1],  # Patient 0
        [2],  # Patient 1
        [0, 2],  # Patient 2
        [1],  # Patient 3
        [0, 1, 2],  # Patient 4
        [0],  # Patient 5
        [1, 2],  # Patient 6
        [2],  # Patient 7
    ]

    # All possible label classes (diagnoses)
    classes = [0, 1, 2]  # e.g., 0=AFib, 1=Tachycardia, 2=Normal

    # Split ratios: 60% train, 20% validation, 20% test
    ratios = [0.6, 0.2, 0.2]

    # Quality scores for each patient (0-4 scale)
    # 0 = poor quality (can't go to clean folds)
    # 1-3 = acceptable quality
    # 4 = excellent quality (always goes to final fold)
    qualities = [2, 3, 1, 2, 3, 0, 2, 3]

    # Number of ECG recordings per patient
    # Some patients might have multiple recordings
    ecgs_per_patient = [5, 3, 4, 2, 6, 3, 4, 3]

    # Number of clean folds (last n folds that require quality > 0)
    # Here, the last fold (test set) will only contain quality > 0 samples
    nr_clean_folds = 1

    print("=" * 60)
    print("STRATIFIED SAMPLING EXAMPLE")
    print("=" * 60)
    print(f"\nTotal patients: {len(data)}")
    print(f"Total ECGs: {sum(ecgs_per_patient)}")
    print(f"Classes: {classes}")
    print(f"Split ratios: {ratios} (train/val/test)")
    print(f"Clean folds: last {nr_clean_folds} fold(s)\n")

    # Perform stratified sampling
    stratified_ids, stratified_labels = stratify(
        data=data,
        classes=classes,
        ratios=ratios,
        qualities=qualities,
        ecgs_per_patient=ecgs_per_patient,
        nr_clean_folds=nr_clean_folds,
    )

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    fold_names = ["Train", "Validation", "Test"]
    for i, (ids, labels) in enumerate(zip(stratified_ids, stratified_labels)):
        print(f"\n{fold_names[i]} Fold:")
        print(f"  Patient IDs: {ids}")
        print(f"  Number of patients: {len(ids)}")
        print(f"  Number of ECGs: {sum(ecgs_per_patient[j] for j in ids)}")
        print(f"  Labels distribution:")

        # Count label occurrences
        label_counts = {c: 0 for c in classes}
        for label_list in labels:
            for label in label_list:
                label_counts[label] += 1

        for label, count in label_counts.items():
            print(f"    Class {label}: {count} patients")

        # Check quality in test fold
        if i == len(ratios) - 1 and nr_clean_folds >= 1:
            print(f"  Quality scores: {[qualities[j] for j in ids]}")
            print(f"  (All should be > 0 for clean fold)")
