
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

def refactor_initial_datasets():
    """
    Only necessary when files have not yet been corrected
    """

    """ MAC dropbox paths """                
    
    # Change tif to jpg
    path = '/Users/justinwhatley/Dropbox/FevensLab/FNAB_raw/dataset_3/G2'
    tif_to_jpg(path)
    path = '/Users/justinwhatley/Dropbox/FevensLab/FNAB_raw/dataset_3/G3'
    tif_to_jpg(path)

    # Change 100x, 400x style to standard a and b
    path = '/Users/justinwhatley/Dropbox/FevensLab/FNAB_raw/dataset_1/MG2/100x'
    rename_with_a_b_style(path, '100x')

    path = '/Users/justinwhatley/Dropbox/FevensLab/FNAB_raw/dataset_1/MG2/400x'
    rename_with_a_b_style(path, '400x')

    path = '/Users/justinwhatley/Dropbox/FevensLab/FNAB_raw/dataset_1/MG3/100x'
    rename_with_a_b_style(path, '100x')

    path = '/Users/justinwhatley/Dropbox/FevensLab/FNAB_raw/dataset_1/MG3/400x'
    rename_with_a_b_style(path, '400x')

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
    Splits all images in the directory to many images of specific height and width. These will be stored
    in a the output_path
    """
    start_num = 1
    
    for root, dirs, files in os.walk(input_path, topdown=False):
        # Crops all files in the directory
        for name in files:
            # Skips files that have already been cropped
            extension = os.path.splitext(os.path.join(root, name))[1].lower().strip() 
            print(extension)
            if 'part' in name: 
                print('failed load')
                continue

            infile = os.path.join(root,name)
            print(infile)
            try: 
                for k, piece in enumerate(crop(infile,height,width), start_num):
                    img = Image.new('RGB', (height,width), 255)
                    img.paste(piece)
                    # TODO change output to jpg
                    new_name = name.split('.')[0] + "_part%s.png" % k
                    output = os.path.join(output_path, new_name)
                    img.save(output)
            except Exception as e:
                print(e)

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

def get_patients(file_list):
    return set([name.split('-')[0] for name in file_list])

def get_images_from_patient(file_list, patient_id):
    return [name for name in file_list if name.split('-')[0] == patient_id]

def trim_bins(bins, bin_size, evenly = False):
    import random

    # If bin size is not set, set it to the smallest bin size
    if not bin_size:
        bin_size = min(map(len, bins))
        # Note: Bin sizes must also be suitable for matrix multiplications
    
    if not evenly:
        # Randomly samples from the bins, removing samples until the size matches the desire bin_size
        for bin in bins:   
            print('Bin size: ' + str(len(bin)))
            while len(bin) > bin_size:
                sample = random.sample(bin, 1)[0]
                bin.remove(sample)

    return bins

def bin_files(file_list, number_of_bins, bin_size, separate_by_patient = True):
    import random
    bins = [[] for i in range(number_of_bins)]

    if separate_by_patient:
        patient_set = get_patients(file_list)
        print(len(patient_set))

        # Randomly select from the list of possible patients, add these to the smallest bin
        while patient_set:
            patient_id = random.sample(patient_set, 1)[0]
            # print(patient_id)
            smallest_bin_size = min(map(len, bins))
            for bin in bins:
                # Add all files from a given patient to the smallest fold
                if len(bin) == smallest_bin_size:
                    images_from_patient = get_images_from_patient(file_list, patient_id)   
                    for image in images_from_patient:
                        bin.append(image)
                    break
            patient_set.remove(patient_id)
        trim_bins(bins, bin_size)
        # Trim the bins to have the same number of files
        # TODO - perhaps cap at an even number like 4000? Evenly distribute removals between patients?

    # Size verification
    for bin in bins:
        print(len(bin))              
    
    else:
        # TODO not implemented
        pass
        # Randomly select from the list of files add it to the smallest bin
        # Remove these files from the original list
        # Trim the bins to have the same number of files

    return bins

def select_folds(file_list, fold_size, number_of_folds = 5):
    """
    Separates the files randomly into different directories while ensuring individual patients do not fall into 
    both the training and validation sets
    """
    bin_size = fold_size
    bins = bin_files(file_list, number_of_folds, bin_size, separate_by_patient = True)
    return bins
    # TODO look into whether this is the correct strategy
    # Assign files in each bin to the training/validation sets

    # Returns a list of lists containing the different bins set


def create_training_and_validation_dir(path, class_keyword_1, class_keyword_2):
    # Iterates through image datasets
    training_set = 'train'
    training_directory = os.path.join(path, training_set)
    mkdir(training_directory)
    MG2_directory = os.path.join(training_directory, class_keyword_1) 
    mkdir(MG2_directory)
    MG3_directory = os.path.join(training_directory, class_keyword_2) 
    mkdir(MG3_directory)

    validation_set = 'validation'
    validation_directory = os.path.join(path, validation_set)
    mkdir(validation_directory)
    MG2_directory = os.path.join(validation_directory, class_keyword_1) 
    mkdir(MG2_directory)
    MG3_directory = os.path.join(validation_directory, class_keyword_2) 
    mkdir(MG3_directory)

def prepare_data_by_class(preprocessed_directory, class_keyword_1, class_keyword_2):
    """

    """
    create_training_and_validation_dir(preprocessed_directory, class_keyword_1, class_keyword_2)
    
    class_1_dir = os.path.join(preprocessed_directory, class_keyword_1)
    class_2_dir = os.path.join(preprocessed_directory, class_keyword_2)
    
    # Loads filelists
    class_1_filelist = sorted([f for f in os.listdir(class_1_dir) if os.path.isfile(os.path.join(class_1_dir, f))])
    class_2_filelist = sorted([f for f in os.listdir(class_2_dir) if os.path.isfile(os.path.join(class_2_dir, f))])

    MG2_count = len(class_1_filelist)
    print(MG2_count)
    MG3_count = len(class_2_filelist)
    print(MG3_count)

    class_1_folds = select_folds(class_1_filelist, 4000)
    class_2_folds = select_folds(class_2_filelist, 700)

    class_list = [class_1_folds, class_2_folds]
    return class_list
    

    

def preprocess_images(preprocessed_directory, MG2, MG3):
    """
    """

    # TODO consider the change in pixel density and how this might change the interpretation by the model when not put to the same scale
    # Splits the images and distributes the split image files into preprocessed MG2 and MG3 directories     

    MG2_directory = os.path.join(preprocessed_directory, 'MG2') 
    mkdir(MG2_directory)
    for dir in MG2:
        split_images(dir, MG2_directory, height, width)  
    
    MG3_directory = os.path.join(preprocessed_directory, 'MG3') 
    mkdir(MG3_directory)
    for dir in MG3:
        split_images(dir, MG3_directory, height, width)  

def prepare_datasets(path, height, width):
    """ 
    Prepares training and validation sets
    """
    print('Loading path: ' + path)

    class_keyword_1 = 'MG2'
    class_keyword_2 = 'MG3'

    # Initializes the preprocessed directory
    preprocessed_data = 'preprocessed'
    preprocessed_directory = os.path.join(path, preprocessed_data)
    mkdir(preprocessed_directory)

    # Gets the file paths for different classes
    MG2 = []
    MG3 = []
    dataset_subdirs = get_subdirectories(path) 
    for dataset_subdir in dataset_subdirs:
        if dataset_subdir.split('/')[-1] == preprocessed_data:
            continue
        class_dirs = get_subdirectories(dataset_subdir)
        for class_dir in class_dirs:
            if class_dir.split('/')[-1] == class_keyword_1:
                MG2.append(class_dir)
            if class_dir.split('/')[-1] == class_keyword_2:
                MG3.append(class_dir)

    # preprocess_images(preprocessed_directory, MG2, MG3)

    class_list = prepare_data_by_class(preprocessed_directory, class_keyword_1, class_keyword_2)
    # TODO move loop to testing function where it'll make more sense
    for _class in class_list:
        for i, fold in enumerate(_class)
            validation_fold = i
            # Move the list of selected files to a training and validation folders
            assign_folds_to_training_and_validation(validation_fold, _class)
            # Run training, validation while keeping average
            model = train()
            model_score = validate()
            
            remove_test_files()
            # remove_files

def assign_folds_to_training_and_validation(validation_bin_int, bins):
    # Make record file of the files that were assigned to the folds
    # Copy files to training directory

    # Copy files to validation directory
    pass



def k_fold_validation(k):
    k = 5
    l = int(len(X) / k)
    mse_total, mae_total = 0, 0
    for i in range(k):
        test_x = X[i*l:(i+1)*l]
        test_y = Y[i*l:(i+1)*l]

        train_x = np.concatenate([X[:i*l], X[(i+1)*l:]])
        train_y = np.concatenate([Y[:i*l], Y[(i+1)*l:]])

        model.fit(train_x, train_y, epochs=15)

        predictions = model.predict(test_x)
        mse, mae = model.evaluate(test_x, test_y)
        mse_total += mse
        mae_total += mae

    mse_avg = mse_total / k
    mae_avg = mae_total / k
    print(mse_avg, mae_avg)

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

if __name__ == "__main__":
    # Linux path
    # test_path = '/home/justin/Dropbox/FevensLab/FNAB_raw/'
    
    # Mac path
    test_path = '/Users/justinwhatley/Dropbox/FevensLab/FNAB_raw'

    height, width = 224, 224

    prepare_datasets(test_path, height, width)
