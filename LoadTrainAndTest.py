import random

from PreProcessImageCaptions import load_doc
import glob


# load training and testing set

def load_set(filename_data_input):
    doc = load_doc(filename_data_input)
    dataset = list()
    # process line by line
    for line in doc.split('\n'):
        # skip empty lines
        if len(line) < 1:
            continue
        # get the image identifier
        identifier = line.split('.')[0]
        dataset.append(identifier)
    return set(dataset)


# load training dataset (6K)
filename = './FlowchartData/Text_Data/TrainingImages.txt'
train = load_set(filename)

print('-------------- Load Training and Testing images -----------------')
print('Dataset: %d' % len(train))

# -------------- Load Images -------------------
# Below path contains all the images
images = './FlowchartData/Images/'
# Create a list of all image names in the directory
img = glob.glob(images + '*.jpg')

# -------------- Load Training Image Names -----------------------

# Below file conatains the names of images to be used in train data
train_images_file = './FlowchartData/Text_Data/TrainingImages.txt'
# Read the train image names in a set
train_images = set(open(train_images_file, 'r').read().strip().split('\n'))
# Create a list of all the training images with their full path names
train_img = []

for i in img:  # img is list of full path names of all images
    if i[len(images):] in train_images:  # Check if the image belongs to training set
        train_img.append(i)  # Add it to the list of train images
print(train_img)
random.shuffle(train_img)
# -------------- Load Test Image Names -----------------------
# Below file contains the names of images to be used in test data
test_images_file = './FlowchartData/Text_Data/TestingImages.txt'
# Read the validation image names in a set# Read the test image names in a set
test_images = set(open(test_images_file, 'r').read().strip().split('\n'))

# Create a list of all the test images with their full path names
test_img = []

for i in img:  # img is list of full path names of all images
    if i[len(images):] in test_images:  # Check if the image belongs to test set
        test_img.append(i)  # Add it to the list of test images

random.shuffle(test_img)

