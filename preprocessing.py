
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

def split_images(path, height, width):
    start_num = 1
    
    for root, dirs, files in os.walk(path, topdown=False):
        for name in files:  
            if 'part' in name:
                continue

            infile = os.path.join(root,name)
            print(infile)
            for k, piece in enumerate(crop(infile,height,width), start_num):
                img = Image.new('RGB', (height,width), 255)
                img.paste(piece)
                new_name = name.split('.')[0] + "_part%s.png" % k
                output_path = os.path.join(path, new_name)
                img.save(output_path)


def prepare_datasets(path, height, width):
    """ 
    Prepares training and validation sets
    """
    split_images(path, height, width)
    
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

test_path = '/home/justin/Dropbox/Fevens Lab/FNAB_raw/test'
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
