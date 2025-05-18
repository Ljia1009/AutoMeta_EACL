

def get_data(path) -> list:
    """ returns a list of dictionaries, each containing 'input_text' and 'target_text' keys """
    path = 'src/models/finetune/data/dev_data.txt'
    with open(path, 'r') as f:
        lines = f.readlines()
    data = []
    for line in lines:
        #         print(line)
        input_text, target_text = line.strip().split('\t')
#         print(input_text, '\n', target_text)
        data.append({'input_text': input_text, 'target_text': target_text})
    return data


# if __name__ == "__main__":
#     data = get_data()
#     print(data[:2])
