import torch
import torchvision.transforms as transforms
import torch.utils.data as data
import os
import pickle
import h5py
import string
import numpy as np
import nltk
from PIL import Image
from build_vocab import Vocabulary


class FlickrDataset(data.Dataset):
    """FlickrDataset Custom Dataset compatible with torch.utils.data.DataLoader."""

    def __init__(self, root, anno_file, vocab, det_file, transform=None):
        """Set the path for images, captions and vocabulary wrapper.

        Args:
            root: image directory.
            json: annotation file path.
            vocab: vocabulary wrapper.
            transform: image transformer.
        """

        self.root = root
        id_caption_dict, self.id_det_index_dict = self.extract_annotation(anno_file)
        self.flickr = id_caption_dict
        self.ids = id_caption_dict.keys()
        self.vocab = vocab
        self.transform = transform
        self.det_features, self.det_features_locs = self._get_det_feat(det_file)

    def extract_annotation(self, filename):
        f = open(filename)
        line = f.readline()
        res = dict()
        id2det = dict()

        counter = 0
        prev_id = ''
        while line != "":
            segs = line.split("\t")
            try:
                img_name = segs[0].split(".")[0] + "_" + segs[0][-1]  # delete ".jpg#"
            except Exception as e:
                print segs
            id, caption = img_name, segs[1]
            res[id] = caption
            id = id.split('_')[0]
            if id != prev_id:
                id2det[id] = counter
                counter += 1
                prev_id = id
            line = f.readline()
        return res, id2det

    def _get_det_feat(self, det_file):
        hf = h5py.File(det_file, 'r')
        return hf['image_features'], hf['spatial_features']

    def __getitem__(self, index):
        """Returns one data pair ( image, caption, image_id )."""
        flickr = self.flickr
        vocab = self.vocab
        ann_id = self.ids[index]
        caption = flickr[ann_id]
        img_id = ann_id
        filename = img_id.split("_")[0] + ".jpg"
        det_index = self.id_det_index_dict[img_id.split('_')[0]]
        det_feat = self.det_features[det_index]  # ndarray, (36, 2048)
        det_feat_loc = self.det_features_locs[det_index]  # ndarray, (36, 6)

        path = filename

        image = Image.open(os.path.join(self.root, path)).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)

        # Convert caption (string) to word ids.
        tokens = str(caption).lower().translate(None, string.punctuation).strip().split()
        caption = []
        caption.append(vocab('<start>'))
        caption.extend([vocab(token) for token in tokens])
        caption.append(vocab('<end>'))
        target = torch.Tensor(caption)

        # add location info into det_features
        det_feat = np.concatenate((det_feat, det_feat_loc), axis=1)
        det_feat = torch.Tensor(det_feat)
        return image, target, det_feat, img_id, filename

    def __len__(self):
        return len(self.ids)


def collate_fn(data):
    """Creates mini-batch tensors from the list of tuples (image, caption).
    
    We should build custom collate_fn rather than using default collate_fn, 
    because merging caption (including padding) is not supported in default.

    Args:
        data: list of tuple (image, caption). 
            - image: torch tensor of shape (3, 256, 256).
            - caption: torch tensor of shape (?); variable length.

    Returns:
        images: torch tensor of shape (batch_size, 3, 256, 256).
        targets: torch tensor of shape (batch_size, padded_length).
        lengths: list; valid length for each padded caption.
        img_ids: image ids in COCO dataset, for evaluation purpose
        filenames: image filenames in COCO dataset, for evaluation purpose
    """

    # Sort a data list by caption length (descending order).
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, captions, det_features, img_ids, filenames = zip(*data)  # unzip

    # Merge images (from tuple of 3D tensor to 4D tensor).
    images = torch.stack(images, 0)
    det_features = torch.stack(det_features, 0)
    img_ids = list(img_ids)
    filenames = list(filenames)

    # Merge captions (from tuple of 1D tensor to 2D tensor).
    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]
    return images, targets, lengths, det_features, img_ids, filenames


def get_loader(root, anno_file, vocab, det_file, transform, batch_size, shuffle, num_workers):
    """Returns torch.utils.data.DataLoader for custom Flickr dataset."""
    # Flickr caption dataset
    Flickr = FlickrDataset(root=root,
                           anno_file=anno_file,
                           vocab=vocab,
                           det_file=det_file,
                           transform=transform)

    # Data loader for COCO dataset
    # This will return (images, captions, lengths) for every iteration.
    # images: tensor of shape (batch_size, 3, 224, 224).
    # captions: tensor of shape (batch_size, padded_length).
    # lengths: list indicating valid length for each caption. length is (batch_size).
    data_loader = torch.utils.data.DataLoader(dataset=Flickr,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers,
                                              collate_fn=collate_fn)
    return data_loader
