
def load_doc(filename_data_input):
    # open the file as read only
    file = open(filename_data_input, 'r')
    # read all text
    text = file.read()
    # close the file
    file.close()
    return text


def load_descriptions(caption_data):
    mapping = dict()
    # process lines
    for line in caption_data.split('\n'):
        # split line by white space
        tokens = line.split()
        if len(line) < 2:
            continue
        # take the first token as the image id, the rest as the description
        image_id, image_desc = tokens[0], tokens[1:]
        # extract filename from image id
        image_id = image_id.split('.')[0]
        # convert description tokens back to the whole string
        image_desc = ' '.join(image_desc)
        # create the list if needed
        if image_id not in mapping:
            mapping[image_id] = list()
        # store description
        mapping[image_id].append(image_desc)
    return mapping


'''
def clean_descriptions(descriptions):
    # prepare translation table for removing punctuation
    table = str.maketrans('', '', string.punctuation)
    for key, desc_list in descriptions.items():
        for i in range(len(desc_list)):
            desc = desc_list[i]
            # tokenize
            desc = desc.split()
            # convert to lower case
            desc = [word.lower() for word in desc]
            # remove punctuation from each token
            desc = [w.translate(table) for w in desc]
            # remove hanging 's' and 'a'
            desc = [word for word in desc if len(word) > 1]
            # remove tokens with numbers in them
            desc = [word for word in desc if word.isalpha()]
            # store as string
            desc_list[i] = ' '.join(desc)

'''


def to_vocabulary(descriptions):
    # build a list of all description strings
    all_desc = set()
    for key in descriptions.keys():
        [all_desc.update(d.split()) for d in descriptions[key]]
    return all_desc


def save_descriptions(descriptions, filename):
    lines = list()
    for key, desc_list in descriptions.items():
        for desc in desc_list:
            lines.append(key + ' ' + desc)
    data = '\n'.join(lines)
    file = open(filename, 'w')
    file.write(data)
    file.close()


filenameDescriptions = "./FlowchartData/Text_Data/Token.txt"
doc = load_doc(filenameDescriptions)
print(doc[:500])

print('------------ Load and Preprocess the image captions ---------------')
# parse descriptions
descriptions_map = load_descriptions(doc)
print('Loaded: %d ' % len(descriptions_map))

print(descriptions_map['00000013'])
# for id in descriptions_map:
# print(list(descriptions_map.keys())[1:5])

vocabulary = to_vocabulary(descriptions_map)
print('Original Vocabulary Size: %d' % len(vocabulary))
print(vocabulary)
save_descriptions(descriptions_map, 'descriptions.txt')
