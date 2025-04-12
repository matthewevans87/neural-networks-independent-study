import random

def split_dataset(input_file, test_file, train_file, test_ratio):
    """
    Splits a text file into test and training datasets based on the given ratio.

    Args:
        input_file (str): Path to the input text file.
        test_file (str): Path to save the test dataset.
        train_file (str): Path to save the training dataset.
        test_ratio (float): Proportion of lines to include in the test dataset (0 < test_ratio < 1).
    """
    with open(input_file, 'r') as file:
        lines = file.readlines()

    random.shuffle(lines)
    split_index = int(len(lines) * test_ratio)

    test_lines = lines[:split_index]
    train_lines = lines[split_index:]

    with open(test_file, 'w') as file:
        file.writelines(test_lines)

    with open(train_file, 'w') as file:
        file.writelines(train_lines)

split_dataset('./data/tinyshakespear.txt', './data/test.txt', './data/train.txt', 0.2)

