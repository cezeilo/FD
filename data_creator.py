import pickle
import os
import string
from fontpreview import FontPreview
import pandas as pd
import requests, zipfile, io
from sklearn.model_selection import train_test_split
import re

font_skip_list = []

#For a given font file, create the alphabet and the numbers 0-9
def create_alphabet(font_file, parent_folder):
    font = FontPreview(font_file)
    font_name = font.font.getname()[0]

    #Loop through all the letters and create their images
    for char in string.ascii_letters:
        font.font_text = char
        font.bg_color = (255, 255, 255)  # white BG
        font.dimension = (512, 512)  # Dimension consistent with the default resolution for diffusion models
        font.fg_color = (0, 0, 0)  # Letter color
        font.set_font_size(300)  # font size ~ 300 pixels
        font.set_text_position('center')  # center placement

        if char in string.ascii_lowercase:
            image_file_name = 'lower_' + char + '_' + font_name + '.jpg'
            save_sub_folder = 'lower_' + char
        else:
            image_file_name = 'upper_' + char + '_' + font_name + '.jpg'
            save_sub_folder = 'upper_' + char

        save_path = os.path.abspath(os.path.join(parent_folder, save_sub_folder))

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        font.save(os.path.abspath(os.path.join(save_path, image_file_name)))


    #Loop through all the digits and create their images
    for num in string.digits:
        font.font_text = num
        font.bg_color = (255, 255, 255)  # white BG
        font.dimension = (512, 512)  # Dimension consistent with the default resolution for diffusion models
        font.fg_color = (0, 0, 0)  # Letter color
        font.set_font_size(300)  # font size ~ 300 pixels
        font.set_text_position('center')  # center placement

        image_file_name = 'number_' + num + '_' + font_name + '.jpg'
        save_sub_folder = 'number_' + num
        save_path = os.path.abspath(os.path.join(parent_folder, save_sub_folder))

        if not os.path.exists(save_path):
            os.makedirs(save_path)


        font.save(os.path.abspath(os.path.join(save_path, image_file_name)))

#Download the zip file, but only extract the requested file. Deletes the zip file when done!
def download_and_extract(df, save_folder):
    link = df['Link']
    requested_file_name = df['Filename']

    r = requests.get(link)
    try:
        z = zipfile.ZipFile(io.BytesIO(r.content))
        z.extract(requested_file_name, path=os.path.abspath(save_folder))
    except:
        pass

#Uses pandas to read through the CSV from sheets without the need of constantly redownloading
def get_fonts(font_file_folder, font_image_folder):
    sheet_url = 'https://docs.google.com/spreadsheets/d/1VB10WsgaWIzCrovDXRroJTSkjH0z4ARo0PgrkAkyOI8/'
    sheet_url += 'export?format=csv&gid=0'
    df = pd.read_csv(sheet_url)


    #head = df.head()

    for i in df.iterrows(): #Yeah yeah this is terribly ineffecient \
                                # but thats when its scaled to over 10k rows. we only have a couple 100
        download_and_extract(i[1], font_file_folder)


    fonts = os.listdir(os.path.abspath(font_file_folder))
    for f in fonts:
        file_path = os.path.abspath(os.path.join(font_file_folder, f))
        create_alphabet(file_path, font_image_folder)
    pass

#Uses pandas to read through the CSV from sheets without the need of constantly redownloading
def get_texts(images_folder, text_folder, fonts_folder):
    sheet_url = 'https://docs.google.com/spreadsheets/d/1VB10WsgaWIzCrovDXRroJTSkjH0z4ARo0PgrkAkyOI8/'
    sheet_url += 'export?format=csv&gid=0'
    df = pd.read_csv(sheet_url)


    #head = df.head()

# Iterate through every row in the dataframe
    for i in df.iterrows():
        row_data = i[1]
        try:
            file_name = FontPreview(os.path.join(fonts_folder, row_data['Filename'])).font.getname()[0]
        except:
            continue

        #For each row, we find the corresponding folder name and write the text data for all classes
            #ie. lower_a, upper_Q, etc.
        files_in_folder = os.listdir(os.path.abspath(images_folder))

        for curr_file in files_in_folder:
            text_file_name = curr_file + '_' + file_name + '.txt'
            sub_folder_path = os.path.abspath(os.path.join(text_folder, curr_file))
            text_file_path = os.path.abspath(os.path.join(sub_folder_path, text_file_name))

            if not os.path.exists(sub_folder_path):
                os.makedirs(sub_folder_path)

            #Extract text description data
            font_characteristics = row_data['Descriptors']
            font_properties = row_data['Weight'] + ' ' + row_data['Courner Rounding'] + ' ' + row_data['Serif'] + ' ' + \
                              row_data['Dynamics'] + ' ' + row_data['Width'] + ' ' + row_data['Capitals']

            # Determine prefix for metadata
            if 'lower' in str(curr_file):  # we know its an upper or lower letter
                prefix = 'A lowercase {} '.format(str(curr_file).replace('lower_', '').split('.')[0])

            if 'upper' in str(curr_file):
                prefix = 'An uppercase {} '.format(str(curr_file).replace('upper_', '').split('.')[0])

            if 'number' in str(curr_file):
                prefix = 'The number {} '.format(str(curr_file).split('.')[0].replace('_', '')[-1])

            #If prefix doesnt have a space..add one 
            if prefix[-1] != ' ':
                prefix = prefix + ' '
            
            if font_characteristics[-1] != ' ':
                font_characteristics = font_characteristics + ' '

            font_text_data = prefix + 'which has traits ' + font_characteristics + 'and properties ' + font_properties

            #Write the data to the file
            with open (text_file_path, 'w') as f:
                f.write(font_text_data)
    pass


#Create the training/testing folders
def create_dataset(images_folder, font_file_path, training_data_path):

    #Step 1: Get all the file names in every subfolder as well as the class_ids

    #1a) Create class IDs
    class_ids = {}
    files_in_folder = os.listdir(os.path.abspath(images_folder))
    for ix, val in enumerate(files_in_folder, start=1):
        class_ids[val] = ix

    #1b) Get all files
    all_files = []
    all_files_class_ids = []
    for path, subdirs, files in os.walk(images_folder):
        for name in files:
            file_name_with_path = os.path.join(path, name)
            file_name = file_name_with_path.split('\\')[-1][:-4]
            file_name_proper_path = os.path.join(file_name_with_path.split('\\')[1], file_name_with_path.split('\\')[-1][:-4])
            all_files.append(file_name_proper_path)

            if 'lower' in str(file_name) or 'upper' in str(file_name):
                label = str(file_name)[:7]
            else:
                label = str(file_name)[:8]
            all_files_class_ids.append(class_ids[label])

    #Step 2: Splt the data according to split ratio
    data_train, data_test, labels_train, labels_test = train_test_split(all_files, all_files_class_ids, test_size=0.10, random_state=42)

    #2b) create training and testing folders
    if not os.path.exists('GALIP/data/fonts/train'):
        os.makedirs('GALIP/data/fonts/train')

    if not os.path.exists('GALIP/data/fonts/test'):
        os.makedirs('GALIP/data/fonts/test')


    #Step 3: Create the filenames.pickle and class_info.pickle files
    #3a) Write pickle files to training folder
    with open(os.path.join('GALIP/data/fonts/train', 'filenames.pickle'), 'wb') as f:
        pickle.dump(data_train, f)

    with open(os.path.join('GALIP/data/fonts/train', 'class_info.pickle'), 'wb') as f:
        pickle.dump(labels_train, f)

    #3b) Write pickle files to test folder
    with open(os.path.join('GALIP/data/fonts/test', 'filenames.pickle'), 'wb') as f:
        pickle.dump(data_test, f)

    with open(os.path.join('GALIP/data/fonts/test', 'class_info.pickle'), 'wb') as f:
        pickle.dump(labels_test, f)

    pass

def create_dataset_2(images_folder, font_file_path, training_data_path):
    font_names = []


    #Step 1: Get all the file names in every subfolder as well as the class_ids

    #1a) Create class IDs
    class_ids = {}
    files_in_folder = os.listdir(os.path.abspath(images_folder))
    for ix, val in enumerate(files_in_folder, start=1):
        class_ids[val] = ix

    #1b) Get all files
    all_files = []
    all_files_class_ids = []
    for path, subdirs, files in os.walk(images_folder):
        for name in files:
            file_name_with_path = os.path.join(path, name)
            file_name = file_name_with_path.split('\\')[-1][:-4]

            lwr = re.findall("^lower_[a-z]_", file_name)

            if lwr:
                font_name = re.sub('^lower_[a-z]_', '', file_name)
        
            upr = re.findall("^upper_[A-Z]_", file_name)

            if upr:
                font_name = re.sub('^upper_[A-Z]_', '', file_name)


            nmbr = re.findall("^number_[0-9]_", file_name)
        
            if nmbr:
                font_name = re.sub('^number_[0-9]_', '', file_name)

            font_names.append(font_name)

            file_name_proper_path = os.path.join(file_name_with_path.split('\\')[1], file_name_with_path.split('\\')[-1][:-4])
            all_files.append(file_name_proper_path)

            if 'lower' in str(file_name) or 'upper' in str(file_name):
                label = str(file_name)[:7]
            else:
                label = str(file_name)[:8]
            all_files_class_ids.append(class_ids[label])

    #Step 2: Splt the FONT data according to split ratio
    font_names = list(set(font_names))
    font_train, font_test = train_test_split(font_names, test_size=0.10, random_state=42)
    
    #   Now that we have the font train and test splits 
    #   Loop through all_files, grab the filename, and see what it belongs
    #   to. 


    data_train = []
    data_test = [] 

    for f in all_files:
        font_name_with_label = str(f.split('\\')[1])
        lwr = re.findall("^lower_[a-z]_", font_name_with_label)

        if lwr:
            font_name = re.sub('^lower_[a-z]_', '', font_name_with_label)
        
        upr = re.findall("^upper_[A-Z]_", font_name_with_label)

        if upr:
            font_name = re.sub('^upper_[A-Z]_', '', font_name_with_label)

         

        nmbr = re.findall("^number_[0-9]_", font_name_with_label)
        
        if nmbr:
            font_name = re.sub('^number_[0-9]_', '', font_name_with_label)


        #Do magic here 
        if font_name in font_train:
            data_train.append(f)
        elif font_name in font_test:
            data_test.append(f)
        else:            
            print(font_name, f)
            raise Exception('ERROR IN FORMATTING FILENAME')
    

    #2b) create training and testing folders
    if not os.path.exists('GALIP/data/fonts/train'):
        os.makedirs('GALIP/data/fonts/train')

    if not os.path.exists('GALIP/data/fonts/test'):
        os.makedirs('GALIP/data/fonts/test')


    #Step 3: Create the filenames.pickle and class_info.pickle files
    #3a) Write pickle files to training folder
    with open(os.path.join('GALIP/data/fonts/train', 'filenames.pickle'), 'wb') as f:
        pickle.dump(data_train, f)

    #with open(os.path.join('GALIP/data/fonts/train', 'class_info.pickle'), 'wb') as f:
    #    pickle.dump(labels_train, f)

    #3b) Write pickle files to test folder
    with open(os.path.join('GALIP/data/fonts/test', 'filenames.pickle'), 'wb') as f:
        pickle.dump(data_test, f)

    #with open(os.path.join('GALIP/data/fonts/test', 'class_info.pickle'), 'wb') as f:
    #    pickle.dump(labels_test, f)

    pass
    


if __name__ == '__main__':
    get_fonts('font_files', 'images')
    get_texts('images', 'text', 'font_files')
    create_dataset('images', 'font_files', 'GALIP/data/fonts/train')
    pass
