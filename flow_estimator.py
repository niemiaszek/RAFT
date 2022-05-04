import sys
sys.path.append('core')

import argparse
import os
import cv2
import glob
import numpy as np
import torch
from PIL import Image

from raft import RAFT
from utils import flow_viz, frame_utils
from utils.utils import InputPadder



DEVICE = 'cuda'

def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)


def viz(img1, img2, flow_up, flow_down):
    img1 = img1[0].permute(1,2,0).cpu().numpy()
    img2 = img2[0].permute(1,2,0).cpu().numpy()
    flow_up = flow_up[0].permute(1,2,0).cpu().numpy()
    flow_down = flow_down[0].permute(1,2,0).cpu().numpy()


    # map flow to rgb image
    flow_up = flow_viz.flow_to_image(flow_up)
    flow_down = flow_viz.flow_to_image(flow_down)
    imgs = np.concatenate([img1, img2], axis=0)
    flows = np.concatenate([flow_up, flow_down], axis=0)
    img_flow = np.concatenate([imgs, flows], axis=1)


    cv2.imshow('image', img_flow[:, :, [2,1,0]]/255.0)
    cv2.waitKey()


def demo(args):
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model))

    model = model.module
    model.to(DEVICE)
    model.eval()

    with torch.no_grad():
        images = glob.glob(os.path.join(args.path, '*.png')) + \
                 glob.glob(os.path.join(args.path, '*.jpg'))
        
        images = sorted(images)
        for imfile1, imfile2 in zip(images[:-1], images[1:]):
            image1 = load_image(imfile1)
            image2 = load_image(imfile2)

            
            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)

            _, flow_up = model(image1, image2, iters=20, test_mode=True)
            _, flow_down =  model(image2, image1, iters=20, test_mode=True)
            flow_up_ = padder.unpad(flow_up[0]).permute(1, 2, 0).cpu().numpy()
            flow_down_ = padder.unpad(flow_down[0]).permute(1, 2, 0).cpu().numpy()
            frame_num = int(imfile1.split("/")[-1].split(".")[0])
            # match forward and backward flow numbers 
            filename_fr = str(frame_num)
            filename_bk = str(frame_num)
            output_file_up = os.path.join("flo_forward_results/", '% s.flo' % filename_fr)
            output_file_down = os.path.join('flo_backward_results/', '% s.flo' % filename_bk)
            frame_utils.writeFlow(output_file_up, flow_up_)
            frame_utils.writeFlow(output_file_down, flow_down_)

            viz(image1, image2, flow_up, flow_down)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint")
    parser.add_argument('--path', help="dataset for evaluation")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    args = parser.parse_args()

    demo(args)
