import string


def remove_punctuation(input_string):
    translator = str.maketrans("", "", string.punctuation)
    clean_string = input_string.translate(translator)
    return clean_string


if __name__ == '__main__':
    print("Opening training data and filtering out sentences with less than 2 words...")
    lines = []
    with open("../../../data/simplebooks/simplebooks-92-raw/train.txt") as file:
        for line in file.readlines():
            line_clean = remove_punctuation(line)
            line_clean = line_clean.replace('\n', "")
            line_clean = line_clean.replace("\n", "")
            line_length = len(line_clean.split(" "))
            if line != "" and line != '\n' and line != "\n" and line_length >= 2:
                lines.append(line)

    print("Saving half of the data for model training.")
    with open("../../../data/simplebooks/train_small.txt", "w+") as file:
        for line in lines:
            file.write(line)
    print("Done!")
