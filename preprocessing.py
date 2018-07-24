
import os
from PIL import Image

def tif_to_jpg(path):

    for root, dirs, files in os.walk(path, topdown=False):
        for name in files:
            print(os.path.join(root, name))
            extension = os.path.splitext(os.path.join(root, name))[1].lower() 
            if extension == ".tiff" or extension == ".tif":
                if os.path.isfile(os.path.splitext(os.path.join(root, name))[0] + ".jpg"):
                    print("A jpeg file already exists for: " + name)
                # If a jpeg is *NOT* present, create one from the tiff.
                else:
                    outfile = os.path.splitext(os.path.join(root, name))[0] + ".jpg"
                    try:
                        im = Image.open(os.path.join(root, name))
                        print("Generating jpeg for %s" % name)
                        im.thumbnail(im.size)
                        im.save(outfile, "JPEG", quality=100)
                    except Exception as e:
                        print(e)

def rename_with_a_b_style(path, magnification):
    """
    Sets file names names to standard 'a' for 100x magnification and 'b' for 400x magnification
    """
    for root, dirs, files in os.walk(path, topdown=False):
        for name in files:  
            print(name)
            name_with_slide_number = name.split('_')[0].split('-')
            
            patient_id = name_with_slide_number[0]
            slide_number = 0
            if len(name_with_slide_number) > 1:
                slide_number = int(name_with_slide_number[1])

            slide_number = str(slide_number+1)
            
            source = os.path.join(path, name)

            if magnification == '100x':
                new_filename = patient_id + '-' + slide_number + 'a' + '.jpg'
                destination = os.path.join(path, new_filename)
                os.rename(source, destination)
                print('Source: ' + source)
                print('Changed to: ' + destination)
                
            if magnification == '400x':
                new_filename = patient_id + '-' + slide_number + 'b' + '.jpg'
                destination = os.path.join(path, new_filename)
                os.rename(source, destination)
                print('Source: ' + source)
                print('Changed to: ' + destination)

# def pair_magnifications():
#     """
#     """
#     pass


def crop(infile, height, width):
    """
    Breaks up an image files into multiple files of a set height and width
    """

    im = Image.open(infile)
    imgwidth, imgheight = im.size
    for i in range(imgheight//height):
        for j in range(imgwidth//width):
            box = (j*width, i*height, (j+1)*width, (i+1)*height)
            yield im.crop(box)

def split_images(input_path, output_path, height, width):
    """
    Splits all images in the 
    """
    start_num = 1
    
    for root, dirs, files in os.walk(input_path, topdown=False):
        # Crops all files in the directory
        for name in files:
            # Skips files that have already been cropped
            if 'part' in name:
                continue

            infile = os.path.join(root,name)
            print(infile)
            for k, piece in enumerate(crop(infile,height,width), start_num):
                img = Image.new('RGB', (height,width), 255)
                img.paste(piece)
                new_name = name.split('.')[0] + "_part%s.png" % k
                output = os.path.join(output_path, new_name)
                img.save(output)

def mkdir(directory):
    """
    Safely make directory
    """
    if not os.path.exists(directory):
        os.makedirs(directory)

def get_subdirectories(directory_path):
    """
    Gets subdirectory path
    """
    return [os.path.join(directory_path, name) for name in os.listdir(directory_path)
            if os.path.isdir(os.path.join(directory_path, name))]

def prepare_datasets(path, height, width):
    """ 
    Prepares training and validation sets
    """
    print('Loading path: ' + path)

    # Iterates through image datasets
    training_set = 'train'
    training_directory = os.path.join(path, training_set)
    mkdir(training_directory)

    validation_set = 'validation'
    validation_directory = os.path.join(validation_set)
    mkdir(validation_directory)

    # Gets the file paths for different classes
    dataset_subdirs = get_subdirectories(path) 
    for dataset_subdir in dataset_subdirs:
        get_subdirectories
        class_dir = dataset_subdir.split('/')[-1]
        print(class_dir)                                                                        

    # # Finds the nested files (e.g., )
    # for root, dirs, files in os.walk(path, topdown=False):
        
    #     for class_dir in dirs: 
    #         class_path = os.path.join(root, class_dir) 
    #         print
    #         print(class_path)
    #         for r, d, f in os.walk(class_path, topdown=False):
    #             for fi in f:
    #                 print(fi)
    #         # split_images(path, training_set, height, width)

def concat_channels():
    " TODO https://stackoverflow.com/questions/43196636/how-to-concatenate-two-layers-in-keras"
    from keras.models import Sequential, Model
    from keras.layers import Concatenate, Dense, LSTM, Input, concatenate
    from keras.optimizers import Adagrad

    first_input = Input(shape=(2, ))
    first_dense = Dense(1, )(first_input)

    second_input = Input(shape=(2, ))
    second_dense = Dense(1, )(second_input)

    merge_one = concatenate([first_dense, second_dense])

    third_input = Input(shape=(1, ))
    merge_two = concatenate([merge_one, third_input])

    model = Model(inputs=[first_input, second_input, third_input], outputs=merge_two)
    model.compile(optimizer=ada_grad, loss='binary_crossentropy',
                metrics=['accuracy'])

test_path = '/home/justin/Dropbox/Fevens Lab/FNAB_raw/'
height, width = 224, 224

prepare_datasets(test_path, height, width)

""" MAC dropbox paths """                
# path = os.getcwd()
# path = '/Users/justinwhatley/Dropbox/Fevens Lab/FNAB_raw/dataset_3/G2'
# path = '/Users/justinwhatley/Dropbox/Fevens Lab/FNAB_raw/dataset_3/G3'
# tif_to_jpg(path)

# path = '/Users/justinwhatley/Dropbox/Fevens Lab/FNAB_raw/dataset_1/MG2/100x'
# rename_with_a_b_style(path, '100x')

# path = '/Users/justinwhatley/Dropbox/Fevens Lab/FNAB_raw/dataset_1/MG2/400x'
# rename_with_a_b_style(path, '400x')

# path = '/Users/justinwhatley/Dropbox/Fevens Lab/FNAB_raw/dataset_1/MG3/100x'
# rename_with_a_b_style(path, '100x')

# path = '/Users/justinwhatley/Dropbox/Fevens Lab/FNAB_raw/dataset_1/MG3/400x'
# rename_with_a_b_style(path, '400x')

