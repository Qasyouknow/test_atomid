# coding:utf-8
import numpy as np
import os
import cv2
import csv
import torch
from torchvision.ops import nms
import torch.nn.functional as F
import pandas as pd


COLOR_LIST = [np.array([0,0,0]).astype('uint8'),    # black represents the background on default
              np.array([255,0,0]).astype('uint8'),
              np.array([0,255,0]).astype('uint8'),
              np.array([0,0,255]).astype('uint8')]

def csv2array(path:str):
    array = []
    with open(path, 'r') as f:
        content = csv.reader(f)
        next(content)
        for line in content:
            array.append(line)
    
    return np.array(array)

def getFileList(path:str, endwith:str):
    iter = os.walk(path)
    path_list = []
    for p, d, filelist in iter:
        for name in filelist:
            if name.endswith(endwith):
                path_list.append(os.path.join(p, name))
                
    path_list.sort()
    return path_list

def getImgList(path:str, endwith:str):
    filelist = getFileList(path, endwith)
    img_list = []
    for path in filelist:
        img = cv2.imread(path, 0)
        img_list.append(img)
    
    return img_list

def getArrayList(path:str, endwith:str='csv'):
    filelist = getFileList(path, endwith)
    array_list = []
    for path in filelist:
        array = csv2array(path)
        array_list.append(array[:,[1,2,-1]].astype('float').astype('int'))
    
    return array_list


def nms_by_diam(out_map, diam=3, gray_th=0.1, iou_th=0.5, device='cuda'):
    out_map = out_map.detach().squeeze()
    out_map[out_map<gray_th] *= 0.0
    cood = torch.nonzero(out_map)
    if cood.size(0) == 0:
        return cood

    scores = out_map[cood[:,0],cood[:,1]]
    
    boxes = torch.zeros(size=(cood.shape[0],4), device=device)
    boxes[:,0] = cood[:,0] - diam
    boxes[:,1] = cood[:,1] - diam
    boxes[:,2] = cood[:,0] + diam
    boxes[:,3] = cood[:,1] + diam
    
    keep = nms(boxes.to(device), scores.to(device), iou_th).long()
    keep_centers = cood[keep,:].long()

    return keep_centers

class EpochScoreRecorder():
    def __init__(self, *names:str):
        self.names = names
        self.recorder = {name:[] for name in names}
        self.means = {name:-1.0 for name in names}
    
    def update(self, *scores:float):
        for name, score in zip(self.names, scores):
            self.recorder[name].append(score)
        return self.recorder
            
    def cal_mean(self):
        for name in self.names:
            self.means[name]  = np.mean(self.recorder[name])
        return self.means

def cal_score(points_1, points_2):
    if len(points_1.shape)<2: points_1 = points_1[None,:]
    if len(points_2.shape)<2: points_2 = points_2[None,:]
    
    x_dis = (points_1[:,0][:,None] - points_2[:,0][None,:]) ** 2
    y_dis = (points_1[:,1][:,None] - points_2[:,1][None,:]) ** 2
    dis = torch.sqrt(x_dis + y_dis)

    rematch_indices = dis[dis.argmin(0), :].argmin(1)
    matches = torch.sum(rematch_indices - torch.arange(dis.size(1), device=dis.device) == 0)

    '''Precision and Recall'''
    precision = matches / max(len(points_1),  1)
    recall = matches / max(len(points_2), 1)
    
    '''Chamfer Distance'''
    cd = torch.mean(dis.min(1)[0]) + torch.mean(dis.min(0)[0])
    
    '''Jaccard Score'''
    jaccard = matches / (len(points_1) + len(points_1) - matches)
    
    '''F1 score'''
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    
    return precision, recall, cd ,jaccard, f1

def visualize(inputs, out_diams, out_maps, points_list, names, gray_th:float=0.1, iou_th:float=0.3, window_size:int=8, save_path=None, device='cuda'):
    if points_list is None:
        points_list = [None] * len(inputs)

    for input_img, out_diam, out_map, name, points in zip(inputs, out_diams, out_maps, names, points_list):
        # Restore input image
        image = input_img.clone().squeeze().detach().cpu()
        image -= torch.min(image)
        image /= torch.max(image)
        image *= 255.0
        
        # Refine predicted average spacing
        diam = torch.floor(out_diam).int().item()
        
        # Local spacial maximum filtering
        out_map, maxpool_indices = F.max_pool2d(out_map.unsqueeze(0), window_size, window_size, return_indices=True)
        out_map = F.max_unpool2d(out_map, maxpool_indices, window_size, window_size)
        
        # Non-maximum suppression
        keep_centers = nms_by_diam(out_map, diam, gray_th, iou_th, device)

        if save_path is not None:
            # Draw predicted output
            mark_map = image.clone().squeeze().detach().cpu().numpy()
            mark_map = cv2.cvtColor(mark_map, cv2.COLOR_GRAY2BGR)
            for center in keep_centers:
                cv2.circle(mark_map, (center[1].item(), center[0].item()), max(1, diam//4), (0, 95, 255), -1)
            cv2.imwrite(os.path.join(save_path, str(name)+"_predict.png"), mark_map.astype('uint8'))
            
            # Draw ground truth
            if points is not None:
                points = points.type(torch.LongTensor)
                gt = image.clone().squeeze().detach().cpu().numpy()
                gt = cv2.cvtColor(gt, cv2.COLOR_GRAY2BGR)
                for center in points:
                    cv2.circle(gt, (center[1].item(), center[0].item()), max(1, diam//4), (0, 255, 95), -1)
                cv2.imwrite(os.path.join(save_path, str(name)+"_gt.png"), gt.astype('uint8'))



            '''
            out_map = out_map / (out_map.max() + 1e-6) * 255.0
            cv2.imwrite(os.path.join(save_path, str(name)+"_map.png"), out_map.astype('uint8'))
            '''
        return keep_centers
        
def lr_scheduler(epoch, optimizer, decay_eff=0.1, decayEpoch=[], is_print:bool=False):
    """Decay learning rate by a factor of lr_decay every lr_decay_epoch epochs"""
    if epoch in decayEpoch:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= decay_eff
            if is_print:
                print('New learning rate is: ', param_group['lr'])
    return optimizer

if __name__ == '__main__':
    entity1 = EpochScoreRecorder()
    entity2 = EpochScoreRecorder()
    entity_list = [[entity1, entity2]] * 3
    print(entity_list)

