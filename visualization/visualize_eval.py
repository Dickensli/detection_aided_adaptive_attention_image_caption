import json
import torch
import torch.nn as nn
import numpy as np
import os
import glob
import pickle
import h5py
from build_vocab import Vocabulary
from torch.autograd import Variable
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
import skimage.transform
import matplotlib.patches as patches
from PIL import Image

# Variable wrapper
def to_var(x, volatile=False):
    '''
    Wrapper torch tensor into Variable
    '''
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, volatile=volatile)


# Show multiple images and caption words
def show_images(images, cols=1, titles=None):
    """Display a list of images in a single figure with matplotlib.
    
    Parameters
    ---------
    images: List of np.arrays compatible with plt.imshow.
    
    cols (Default = 1): Number of columns in figure (number of rows is 
                        set to np.ceil(n_images/float(cols))).
    
    titles: List of titles corresponding to each image. Must have
            the same length as titles.
            
    Adapted from https://gist.github.com/soply/f3eec2e79c165e39c9d540e916142ae1
    """

    assert ((titles is None) or (len(images) == len(titles)))

    n_images = len(images)
    if titles is None:
        titles = ['Image (%d)' % i for i in range(1, n_images + 1)]

    fig = plt.figure(figsize=(15, 15))
    for n, (image, title) in enumerate(zip(images, titles)):

        a = fig.add_subplot(np.ceil(n_images / float(cols)), cols, n + 1)
        if image.ndim == 2:
            plt.gray()

        plt.imshow(image)
        a.axis('off')
        a.set_title(title, fontsize=200)

    fig.set_size_inches(np.array(fig.get_size_inches()) * n_images)

    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    plt.show()
    
#  evaluation data loader
class FlickrEvalLoader(datasets.ImageFolder):
    def __init__(self, root, ann_path, det_file, transform=None, target_transform=None,
                 loader=datasets.folder.default_loader):
        '''
        Customized Flickr loader to get Image ids and Image Filenames
        root: path for images
        ann_path: path for the annotation file (e.g., caption_val2014.json)
        '''
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        self.imgs = self.extract_images(ann_path)
        self.det_features, self.det_features_locs = self._get_det_feat(det_file)
        self.det_loc = self._get_det_loc(det_file)

    def extract_images(self, filename):
        f = open(filename)
        line = f.readline()
        res = list()
        self.id2detidx = {}

        pre_img_name = ''
        counter = 0
        while line != "":
            segs = line.split("\t")
            try:
                img_name = segs[0].split(".")[0]
            except Exception as e:
                print segs
            if img_name not in res:
                res.append(img_name)
            if img_name != pre_img_name:
                self.id2detidx[img_name] = counter
                counter += 1
                pre_img_name = img_name
            line = f.readline()
        return res

    def _get_det_feat(self, det_file):
        hf = h5py.File(det_file, 'r')
        return hf['image_features'], hf['spatial_features']
    
    def _get_det_loc(self, det_file):
        hf = h5py.File(det_file, 'r')
        return hf['spatial_features']
    
    def get_image(self, img_id):
        filename = img_id + ".jpg"
        print filename
        path = os.path.join(self.root, filename)
        return filename, self.loader(path)

    def __getitem__(self, index):
        img_id = self.imgs[index]
        
        det_feat = self.det_features[self.id2detidx[img_id]]
        det_feat_loc = self.det_features_locs[self.id2detidx[img_id]]
        det_feat = np.concatenate((det_feat, det_feat_loc), axis=1)
        #det_feat = torch.Tensor(det_feat)
        
        det_l = self.det_loc[self.id2detidx[img_id]]
        filename = img_id + ".jpg"
        path = os.path.join(self.root, filename)

        img = self.loader(path)
        if self.transform is not None:
            fit_img = self.transform(img)       

        return fit_img, img_id, filename, det_feat, det_l

def detect_location(st_x, st_y, w, h, a):
    st_x, st_y, w, h = 256*st_x, 256*st_y, 256*w, 256*h
    st_x = max(0, st_x - (256-a)/2)
    st_y = max(0, st_y - (256-a)/2)
    w = min(w, a - st_x)
    h = min(h, a - st_y)
    return st_x, st_y, w, h
        
def Flickr_visual(model, args, epoch):
    '''
    model: trained model to be evaluated
    args: pre-set parameters
    '''
    model.eval()

    # Validation images are required to be resized to 224x224 already
    transform = transforms.Compose([
        transforms.CenterCrop((args.crop_size, args.crop_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))])

    # Load the vocabulary
    with open(args.vocab_path, 'rb') as f:
        vocab = pickle.load(f)

    # Wrapper the COCO VAL dataset
    fl = FlickrEvalLoader(args.image_dir, args.caption_val_path, args.det_val_file, transform)
    eval_data_loader = torch.utils.data.DataLoader(
        dataset=fl,
        batch_size=args.eval_size,
        shuffle=False, num_workers=args.num_workers,
        drop_last=False)

    # Generated captions to be compared with GT
    print '---------------------Start evaluation on MS-COCO dataset-----------------------'
    count = 0

    for i, (images, image_ids, _, det_features, det_locs) in enumerate(eval_data_loader):
        if count < 150:
            count += 1
            continue
        if count >= 200:
            break
            
        images = to_var(images)
        det_features = to_var(det_features)
        generated_captions, attentions, _ = model.sampler(images, det_features)

        if torch.cuda.is_available():
            captions = generated_captions.cpu().data.numpy()
            attentions = attentions.cpu().data.numpy()
        else:
            captions = generated_captions.data.numpy()
            attentions = attentions.data.numpy()
        
        # Build caption based on Vocabulary and the '<end>' token
        for image_idx in range(captions.shape[0]):
            count += 1
            
            #generate the image
            filename, img = fl.get_image(image_ids[image_idx])
            crop_rectangle = ((256 - args.crop_size)/2, (256 - args.crop_size)/2, (256 + args.crop_size)/2, (256 + args.crop_size)/2)
            img = img.crop(crop_rectangle)
            
            f = plt.figure(figsize=(10,30))
            ax = f.add_subplot(7, 3, 1)
            ax.imshow(img)
            ax.axis('off')
            
            #generate attention
            atten = attentions[image_idx]
            grid_att = atten[ : , :49 ]
            det_att = atten[ : , 49: ]
            det_loc = det_locs[image_idx]
            
            #generate the sentense
            sampled_ids = captions[image_idx]
            sampled_caption = []

            for t in range(len(sampled_ids)):
                word = vocab.idx2word[sampled_ids[t]]
                if word == '<end>':
                    break
                else:
                    sampled_caption.append(word)
                    
                ax = f.add_subplot(7, 3, t+2)                   
                #generate fixed-location attention    
                ax.text(0, 1, '%s'%(word) , color='black', backgroundcolor='white', fontsize=8)
                ax.imshow(img)
                grid_curr = grid_att[t,:].reshape(7,7)
                grid_img = skimage.transform.pyramid_expand(grid_curr, upscale=32, sigma=20)
                ax.imshow(grid_img, alpha=0.6)
                ax.axis('off')
                
                #generate detection-based attention
                det_att_t = det_att[t, : ].tolist()
                bench1 = np.max(grid_att[t, :])
                bench2 = 2 * np.max(grid_att[t, :])
                for s in range(len(det_att_t)):
                    if det_att_t[s] > bench2:
                        loc = det_loc[s, : ]
                        st_x, st_y, w, h = detect_location(loc[0], loc[1], loc[4], loc[5], args.crop_size)
                        rect = patches.Rectangle((st_x,st_y),
                                                 w, h,
                                                 linewidth=1,edgecolor='r',facecolor='none')
                        # Add the patch to the Axes
                        ax.add_patch(rect)
                        ax.axis('off')
                    elif det_att_t[s] > bench1:
                        loc = det_loc[s, : ]
                        st_x, st_y, w, h = detect_location(loc[0], loc[1], loc[4], loc[5], args.crop_size)
                        rect = patches.Rectangle((st_x,st_y),
                                                 w, h,
                                                 linewidth=1,edgecolor='y',facecolor='none')
                        # Add the patch to the Axes
                        ax.add_patch(rect)
                        ax.axis('off')  
            
            plt.savefig("./data/imgs/" + filename + ".jpg")       
            plt.show()
            sentence = ' '.join(sampled_caption)
            print sentence
            
            

           