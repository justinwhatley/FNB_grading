
import os
from PIL import Image

def tif_to_jpg(path):
    # TODO consider converting to lossless png
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
                    new_name = name.split('.')[0] + "_part%s.jpg" % k
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

def remove_dir(path):
    """
    Removes directory, ignoring possible errors
    """
    from shutil import rmtree
    print('Removing directory: ' + path)
    try: 
        rmtree(path)
    except:
        pass


def create_training_and_validation_dir(path, class_list):
    """
    Creates training and validation directories if they do not exist
    """    
    training_set = 'training'
    training_directory = os.path.join(path, training_set)
    mkdir(training_directory)
    for class_name in class_list:
        mkdir(os.path.join(training_directory, class_name))
    
    validation_set = 'validation'
    validation_directory = os.path.join(path, validation_set)
    mkdir(validation_directory)
    for class_name in class_list:
        mkdir(os.path.join(validation_directory, class_name))

def separate_into_k_folds(number_of_folds, class_file_list, files_per_fold):
    """
    Class_file_list now contains both MG2 at index 0 and MG3 at index 1 separated into k folds
    # with bin sizes of the specified file limit per fold
    """

    # Bins the data into k folds
    binned_class_file_list = []
    for class_files_list in class_file_list:
        binned_class_file_list.append(bin_files(class_files_list, number_of_folds))
    
    # Trims the data assignment to create folds of equal size
    for i, binned_file_list in enumerate(binned_class_file_list): 
        binned_file_list = trim_bins(binned_file_list, files_per_fold[i])

    return binned_class_file_list

def bin_files(file_list, number_of_bins, separate_by_patient = True):
    """
    Separates the files randomly into different directories while ensuring individual patients do not fall into 
    both the training and validation sets
    """
    import random
    bins = [[] for i in range(number_of_bins)]

    # Simple assignment of files from each patient to bin with the fewest slides
    if separate_by_patient:
        patient_set = get_patients(file_list)
        # print(len(patient_set))

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
    
    else:
        # TODO not implemented
        pass
        # Randomly select from the list of files add it to the smallest bin
        # Remove these files from the original list
        # Trim the bins to have the same number of files

    return bins

# def select_folds(file_list, fold_size, number_of_folds = 5):
#     """
#     Separates the files randomly into different directories while ensuring individual patients do not fall into 
#     both the training and validation sets
#     """
#     bin_size = fold_size
#     bins = bin_files(file_list, number_of_folds, bin_size, separate_by_patient = True)
#     return bins
#     # TODO look into whether this is the correct strategy
#     # Assign files in each bin to the training/validation sets

#     # Returns a list of lists containing the different bins set

def get_data_by_class(preprocessed_directory, classes_list):
    """
    Note: Currently harcoded to take only two classes and 
    """
 
    class_1_dir = os.path.join(preprocessed_directory, classes_list[0])
    class_2_dir = os.path.join(preprocessed_directory, classes_list[1])
    
    # Loads filelists
    class_1_filelist = sorted([f for f in os.listdir(class_1_dir) if os.path.isfile(os.path.join(class_1_dir, f))])
    class_2_filelist = sorted([f for f in os.listdir(class_2_dir) if os.path.isfile(os.path.join(class_2_dir, f))])

    class_list = [class_1_filelist, class_2_filelist]
    return class_list

def copy_original_data_by_class(preprocessed_directory, subdirectory, class_keyword, raw_data_directory_list):
    """
    Copies the original to the preprocessed data from raw directory
    """
    from distutils.dir_util import copy_tree
    for original_directory in raw_data_directory_list:
        copy_tree(original_directory, os.path.join(preprocessed_directory, subdirectory, class_keyword))

def make_image_patches(preprocessed_directory, subdirectory, class_keyword, class_file_list, height, width):
    """
    Splits the images into patches and distributes these into a directory of that class
    """
    
    patch_directory_name = subdirectory

    directory = os.path.join(preprocessed_directory, patch_directory_name, class_keyword) 
    mkdir(directory)
    for dir in class_file_list:
        split_images(dir, directory, height, width)  

def create_preprocessed_directory(classes_list, class_files_list, preprocessed_directory_path, height, width, overwrite_previous_preprocessed_data = True):
    """
    If the preprocessing directory does not exist, creates one and adds patched data. If overwrite is set, it will remove the previous 
    directory and make a new preprocessed directory
    """
    # Check if the preprocessed directory is empty
    empty_directory = False
    mkdir(preprocessed_directory_path)
    if [f for f in os.listdir(preprocessed_directory_path) if not f.startswith('.')] == []:
        empty_directory = True

    # Prepare preprocessed directory
    if empty_directory:
        print('Creating preprocessed directory: ' + preprocessed_directory_path)
        for i, class_keyword in enumerate(classes_list):
            # Adding preprocessed patches
            make_image_patches(preprocessed_directory_path, 'patched_data', class_keyword, class_files_list[i], height, width)
            # Copying raw data to preprocessed directory
            copy_original_data_by_class(preprocessed_directory_path, 'original_data', class_keyword, class_files_list[i])

    # Remove old preprocessed directory and prepare a new one
    elif not empty_directory and overwrite_previous_preprocessed_data:
        from shutil import rmtree
        print('Removing old preprocessed directory...')
        rmtree(preprocessed_directory_path)
        print('Creating preprocessed directory: ' + preprocessed_directory_path)
        # TODO consider the change in pixel density and how this might change the interpretation by the model when not put to the same scale
        for i, class_keyword in enumerate(classes_list):
            # Adding preprocessed patches
            make_image_patches(preprocessed_directory_path, 'patched_data', class_keyword, class_files_list[i], height, width)
            # Copying raw data to preprocessed directory
            copy_original_data_by_class(preprocessed_directory_path, 'original_data', class_keyword, class_files_list[i])

def get_raw_file_list(raw_data_path, classes_list):
    print('Loading raw data from path: ' + raw_data_path)

    # Gets the file paths for different classes
    class_files_list = [[] for i in range(len(classes_list))]
    dataset_subdirs = get_subdirectories(raw_data_path) 
    for dataset_subdir in dataset_subdirs:
        class_dirs = get_subdirectories(dataset_subdir)
        for class_dir in class_dirs:
            directory_name = class_dir.split('/')[-1]
            # Gets the index associated to the class which was found in the directory names (assumes only one of any specifed class_keyword)
            
            index_list = [i for i, s in enumerate(classes_list) if directory_name in s]
            if index_list:
                index = index_list[0]
            else:
                continue
            # Adds the directory name for files of a given class to the list
            class_files_list[index].append(class_dir)

    return class_files_list

def copy_files(source_directory, destination_directory, filenames):
    """
    Copies all files in a file list from the source directory to a destination directory
    """
    from shutil import copy2
    
    print('Taking files from input directory: ' + source_directory)
    print('Storing them in output directory: ' + destination_directory)

    for filename in filenames:
        copy2(os.path.join(source_directory, filename), destination_directory)

def assign_folds_to_training_and_validation(preprocessed_path, training_validation_path, classes_list, files_in_folds, validation_fold, type = 'original_data'):

    # Appends the type of file to the training_validation directory
    training_validation_path = os.path.join(training_validation_path, type)

    # Create training and validation directories if they do not exist
    create_training_and_validation_dir(training_validation_path, classes_list)

    # TODO Make record file of the files that were assigned to the folds

    # Move training and validation files to the correct directories
    for i, _class in enumerate(files_in_folds): 
        training_list = []
        validation_list = []
        for j, fold in enumerate(_class):
            if j == validation_fold: 
                validation_list = fold
            else: 
                training_list.extend(fold)
        input_directory = os.path.join(preprocessed_path, type, classes_list[i])  
        output_directory = os.path.join(training_validation_path)
 
        # Copy files to training directory  
        copy_files(input_directory, os.path.join(output_directory, 'training', classes_list[i]), training_list)
        # Copy files to validation directory
        copy_files(input_directory, os.path.join(output_directory, 'validation', classes_list[i]), validation_list)