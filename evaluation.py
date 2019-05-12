import numpy as np
import torch

class EvaluationMetrics:
    def __init__(self):
        self.relu = torch.nn.ReLU()

    def average_precision_score(self, bbox, bbox_pred, threshold):
        iou_scores = self.iou_loss(bbox, bbox_pred)
        y_true = np.ones(bbox.size()[0])
        y_loss = self.iou_loss(bbox, bbox_pred).detach().numpy()
        y_score = [1 if v>threshold else 0 for v in y_loss]
        score = np.mean(y_score)
        return(score)
    
    def iou_loss(self, bbox, bbox_pred):
        area1 = (bbox[:, 0, 2] - bbox[:, 0, 0])*(bbox[:, 0, 3] - bbox[:, 0, 1])
        area2 = (bbox_pred[:, 0, 2] - bbox_pred[:, 0, 0]) * \
            (bbox_pred[:, 0, 3] - bbox_pred[:, 0, 1])
        area_intersection = (torch.min(bbox[:, 0, 2], bbox_pred[:, 0, 2]) - torch.max(bbox[:, 0, 0], bbox_pred[:, 0, 0]))*(
            torch.min(bbox[:, 0, 3], bbox_pred[:, 0, 3]) - torch.max(bbox[:, 0, 1], bbox_pred[:, 0, 1]))

        loss = (area_intersection + 1e-4) / \
            (area1 + area2 - area_intersection + 1e-4)
        loss = self.relu(loss)
        #loss = torch.mean(loss, dim=0)
        #loss = 1 - loss
        return(loss)

    def mean_average_precision(self, bbox, bbox_pred):
        scores = 0
        count = 0
        for threshold in range(0, 11):
            scores += self.average_precision_score(bbox = bbox, bbox_pred = bbox_pred, threshold = threshold*0.1)
            count += 1
        scores /= count
        return(scores)

if __name__ == '__main__':
    eval1 = EvaluationMetrics()
    bbox = torch.tensor([[1,2,3,4], [1,2,3,4]], dtype = torch.float).view(-1, 1, 4)
    bbox_pred = torch.tensor([[1,2,3,4], [1,2,300,5]], dtype=torch.float).view(-1, 1, 4)
    y_true = [1,1,1,1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    y_score = [1,0,0,0,1, 0, 1, 1, 0, 0, 0, 0, 0]
    print(eval1.mean_average_precision(bbox = bbox, bbox_pred= bbox_pred))