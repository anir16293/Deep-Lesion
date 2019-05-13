import numpy as np
from evaluation import EvaluationMetrics
from dataloader import ImageDataset
from panet import PanNet
import torch
from torch.utils.data import DataLoader
import pandas as pd
import pickle


csv_path = "/Users/aniruddha/Downloads/DL_info.csv"
Image_slices_dir = "/Users/aniruddha/Downloads/Test"

df = pd.read_csv(csv_path)          # The DL_info.csv file path
df.sort_values("File_name", inplace=True)
df.drop_duplicates(subset="File_name",
                   keep=False, inplace=True)

new_df = df[df['Train_Val_Test'] == 3]


model = PanNet()
model.load_state_dict(torch.load('/Users/aniruddha/Downloads/panet_model_4.dms'))

eval1 = EvaluationMetrics()

map_scores_list = []
eval1 = EvaluationMetrics()
for i in range(1, 9):
    batch_no = 0
    train_dataset = ImageDataset(
        root_dir="/Users/aniruddha/Downloads/Test", dataset_type=3)
    # Set the new_df yourself
    print("Original length is", len(new_df.index))
    loop_df = new_df[new_df['Coarse_lesion_type'] == i]

    train_dataset.df = loop_df

    batch_size = 3
    dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size)

    bbox_global = None
    bbox_pred_global = None

    print(len(dataloader))

    start = True
    count = 0
    for batch in dataloader:
        img = batch['image']
        bbox = batch['lesions']
        bbox_pred = model(img)
        count += 1
        if start:
            bbox_global = bbox
            bbox_pred_global = bbox_pred
            start = False
        else:
            bbox_global = torch.cat((bbox_global, bbox), dim=0)
            bbox_pred_global = torch.cat((bbox_pred_global, bbox_pred), dim=0)
        print(count)

    map_scores_list.append(eval1.mean_average_precision(
        bbox=bbox_global, bbox_pred=bbox_pred_global))

pickle.dump(map_scores_list, open('map_scores_list.p','wb'))