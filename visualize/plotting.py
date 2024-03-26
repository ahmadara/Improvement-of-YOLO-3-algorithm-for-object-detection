from colorama import Fore
import cv2
from google.colab.patches import cv2_imshow
import torch
import numpy as np 
import random
from PIL import Image
from matplotlib import pyplot as plt
import os
def xywh2xyxy(x):
    # Convert bounding box format from [x, y, w, h] to [x1, y1, x2, y2]
    y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2
    y[:, 1] = x[:, 1] - x[:, 3] / 2
    y[:, 2] = x[:, 0] + x[:, 2] / 2
    y[:, 3] = x[:, 1] + x[:, 3] / 2
    return y

def plot_box(bboxes, img,LABELS ,  id = None, color=None, line_thickness=None):

    img = img.permute(0,2,3,1).contiguous()[0].numpy() if isinstance(img, torch.Tensor) else img# [C,H,W] ---> [H,W,C]
    img_size, _, _ = img.shape
    bboxes[:, :4] = xywh2xyxy(bboxes[:, :4])
    tl = line_thickness or round(0.002 * max(img.shape[0:2])) + 1  # line thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    for i, x in enumerate(bboxes):
        c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
        cv2.rectangle(img, c1, c2, color, thickness=tl)
        label = LABELS[int(x[4])]
        if label:
            tf = max(tl - 1, 1)  # font thickness
            t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
            c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
            cv2.rectangle(img, c1, c2, color, -1)  # filled
            cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [0, 0, 0], thickness=tf, lineType=cv2.LINE_AA)

    img = cv2.cvtColor(img* 255.0, cv2.COLOR_RGB2BGR).astype(np.uint8)
    pil_image = Image.fromarray(img)


    return(pil_image )

def visulaize_input(train_dataloaer,DATA,result_directory):
  (img, label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes) = next(iter(train_dataloaer ))
        
  """       
          :param label_sbbox: Small detection layer's label. The size of value is for original image size.
                      shape is [bs, grid, grid, anchors, x+y+w+h+conf+mix+cls_20]
          :param label_mbbox: Same as label_sbbox.
          :param label_lbbox: Same as label_sbbox.
          :param sbboxes: Small detection layer bboxes.The size of value is for original image size.
                          shape is [bs, 150, x+y+w+h]
          :param mbboxes: Same as sbboxes.
          :param lbboxes: Same as sbboxes
  """

  print(Fore.BLUE,'number of classes in dataset: ',len(DATA['CLASSES']))
  print(Fore.MAGENTA,'....intput image and label info are as follows....')
  print(Fore.GREEN,'img.shape: ',Fore.RED,img.shape)
  print(Fore.GREEN,'label_sbbox.shape: ',Fore.RED,label_sbbox.shape)
  print(Fore.GREEN,'label_mbbox.shape: ',Fore.RED,label_mbbox.shape)
  print(Fore.GREEN,'label_lbbox.shape: ',Fore.RED,label_lbbox.shape)
  print(Fore.GREEN,'sbboxes.shape: ',Fore.RED,sbboxes.shape)
  print(Fore.GREEN,'mbboxes.shape: ',Fore.RED,mbboxes.shape)
  print(Fore.GREEN,'lbboxes.shape: ',Fore.RED,lbboxes.shape)
    
  print(Fore.RED,'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
  print(Fore.CYAN,'data augmention has been used in training dataset ')
  print(Fore.CYAN,'two images are merged in some of the following plot ')
  print(Fore.RED,'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
  fig, axs = plt.subplots(4,2,figsize=(30,30))
  for c in range(img.shape[0]):
    if c<8:
      label_sbbox_c  =  torch.unsqueeze(label_sbbox[c,:,:,:],0)
      label_mbbox_c  =  torch.unsqueeze(label_mbbox[c,:,:,:], 0)
      label_lbbox_c  =  torch.unsqueeze(label_lbbox[c,:,:,:],0)
      img_c          =  torch.unsqueeze(img[c,:,:,:],0)
      labels_c = np.concatenate([label_sbbox_c.reshape(-1, 26), label_mbbox_c.reshape(-1, 26),
                              label_lbbox_c.reshape(-1, 26)], axis=0)
      labels_mask_c = labels_c[..., 4]>0
      labels_c = np.concatenate([labels_c[labels_mask_c][..., :4], np.argmax(labels_c[labels_mask_c][..., 6:],
                              axis=-1).reshape(-1, 1)], axis=-1)
      
      img_f = plot_box(labels_c, img_c,LABELS =DATA["CLASSES"], id=1) 
      plt.subplot(4,2,c+1)
      plt.tight_layout()
            
      plt.imshow(img_f, cmap="gray",interpolation='none')

  save_input = os.path.join(result_directory, 'input_image' + '.jpg')
  plt.savefig(save_input)