
import os

def tif_to_jpg(path):
    from PIL import Image

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

