import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

image_directory = '/Users/aniruddha/Downloads/Key_slices'
info_file = '/Users/aniruddha/Downloads/DL_info.csv'

class CTImage:
    def __init__(self, image_directory, info_file, study_type = 'all', patient_metadata = False):
        self.image_directory = image_directory
        self.info_file = pd.read_csv(info_file)
        if study_type != 'all':
            self.info_file = self.info_file[self.info_file.Study_index == study_type]
        self.patient_metadata = patient_metadata
        
    def get_by_user(self, user_idx):
        file_subset = self.info_file[self.info_file.Patient_index == user_idx]
        image_list = file_subset.File_name.values
        if self.patient_metadata:
            gender_list = file_subset.Patient_gender.values
            age_list = file_subset.Patient_age.values
        images = []
        if not self.patient_metadata:
            for image_name in image_list:
                image_path = os.path.join(self.image_directory, image_name)
                images.append(plt.imread(image_path))
        if self.patient_metadata:
            return((images, gender_list, age_list))
        else:
            return((images,))
    
    def get_val_by_lesion(self, lesion_type = 'bone'):
        study_dict = {'bone':1, 'abdomen':2, 'mediastinum': 3,'liver':4, 'lung':5, 'kidney': 6, 'soft_tissue':7, 'pelvis':8}
        val_file = self.info_file[self.info_file.Train_Val_Test == 2]
        file_subset = images_list = val_file[val_file.Coarse_lesion_type == study_dict[lesion_type]]
        images_list = file_subset.File_name.values
        if self.patient_metadata:
            gender_list = file_subset.Patient_gender.values
            age_list = file_subset.Patient_age.values
        images = []
        for image_name in images_list:
            image_path = os.path.join(self.image_directory, image_name)
            images.append(plt.imread(image_path))
        if self.patient_metadata:
            return((images, gender_list, age_list))
        else:
            return((images,))
    
    def get_test_by_lesion(self, lesion_type = 'bone'):
        study_dict = {'bone':1, 'abdomen':2, 'mediastinum': 3,'liver':4, 'lung':5, 'kidney': 6, 'soft_tissue':7, 'pelvis':8}
        test_file = self.info_file[self.info_file.Train_Val_Test == 3]
        file_subset = test_file[test_file.Coarse_lesion_type == study_dict[lesion_type]]
        images_list = file_subset.File_name.values
        if self.patient_metadata:
            gender_list = file_subset.Patient_gender.values
            age_list = file_subset.Patient_age.values
        images = []
        for image_name in images_list:
            image_path = os.path.join(self.image_directory, image_name)
            images.append(plt.imread(image_path))
        if self.patient_metadata:
            return((images, gender_list, age_list))
        else:
            return((images,))

    def get_train(self):
        file_subset = self.info_file[self.info_file.Train_Val_Test == 1]
        images_list = file_subset.File_name.values
        images = []
        if self.patient_metadata:
            gender_list = file_subset.Patient_gender.values
            age_list = file_subset.Patient_age.values
        for image_name in images_list:
            image_path = os.path.join(self.image_directory, image_name)
            images.append(plt.imread(image_path))
        if self.patient_metadata:
            return((images, gender_list, age_list))
        else:
            return((images,))

if __name__ == '__main__':
    ct_images = CTImage(image_directory, info_file, patient_metadata = False)
    plt.imshow(ct_images.get_test_by_lesion('bone')[0][1]) #Visualize the second CT image amongst bone lesion CTs
    plt.show()
