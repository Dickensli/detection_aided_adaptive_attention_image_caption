import math
import json
import argparse
import torch
import torch.nn as nn
import numpy as np
import os
import pickle
from data_loader import get_loader
from adaptive import Encoder2Decoder
from build_vocab import Vocabulary
from torch.autograd import Variable
from torchvision import transforms
from torch.nn.utils.rnn import pack_padded_sequence
from visualize_eval import Flickr_visual

def main(args):
    # To reproduce training results
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # Create model directory
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    # Image Preprocessing
    # For normalization, see https://github.com/pytorch/vision#models
    transform = transforms.Compose([
        transforms.RandomCrop(args.crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))])

    # Load vocabulary wrapper.
    with open(args.vocab_path, 'rb') as f:
        vocab = pickle.load(f)

    # Build training data loader
    data_loader = get_loader(args.image_dir, args.caption_path, vocab, args.det_file,
                             transform, args.batch_size,
                             shuffle=True, num_workers=args.num_workers)

    # Load pretrained model or build from scratch
    adaptive = Encoder2Decoder(args.embed_size, len(vocab), args.hidden_size)

    if args.pretrained:

        adaptive.load_state_dict(torch.load(args.pretrained))
        # Get starting epoch #, note that model is named as '...your path to model/algoname-epoch#.pkl'
        # A little messy here.
        start_epoch = int(args.pretrained.split('/')[-1].split('-')[1].split('.')[0]) + 1

    else:
        start_epoch = 1

    # Constructing CNN parameters for optimization, only fine-tuning higher layers
 
    ch=list(adaptive.encoder.resnet_conv.children())
    #for i in range(len(ch)):
    #    print i,'th:',ch[i]
    #cnn_subs = list(adaptive.encoder.resnet_conv.children())[args.fine_tune_start_layer:]
  
   
    cnn_subs = list(adaptive.encoder.resnet_conv.children())
    
    cnn_params = [list(sub_module.parameters()) for sub_module in cnn_subs]

    #print  "cnn_params",cnn_params
    cnn_params = [item for sublist in cnn_params for item in sublist]

    #print "cnn_params",cnn_params
    #www
    cnn_optimizer = torch.optim.Adam(cnn_params, lr=args.learning_rate_cnn,
                                     betas=(args.alpha, args.beta))

    # Other parameter optimization
    params = list(adaptive.encoder.affine_a.parameters()) + list(adaptive.encoder.affine_b.parameters())+ list(adaptive.decoder.parameters())

    # Will decay later    
    learning_rate = args.learning_rate

    # Language Modeling Loss
    LMcriterion = nn.CrossEntropyLoss()

    # Change to GPU mode if available
    if torch.cuda.is_available():
        adaptive.cuda()
        LMcriterion.cuda()

    # Train the Models
    total_step = len(data_loader)

    cider_scores = []
    best_cider = 0.0
    best_epoch = 0

    adaptive.load_state_dict(torch.load(args.checkpoint_file))
    # Start Training 
    for epoch in range(start_epoch, args.num_epochs + 1):
        if epoch > start_epoch:
            break
    
        # Evaluation on validation set        
        Flickr_visual(adaptive, args, epoch)
    

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_file', type=str, default='./models/adaptive-20.pkl',
                        help='path to retrieve trained models')
    parser.add_argument('-f', default='self', help='To make it runnable in jupyter')
    parser.add_argument('--model_path', type=str, default='./models',
                        help='path for saving trained models')
    parser.add_argument('--crop_size', type=int, default=224,
                        help='size for randomly cropping images')
    parser.add_argument('--vocab_path', type=str, default='./data/vocab.pkl',
                        help='path for vocabulary wrapper')
    parser.add_argument('--image_dir', type=str, default='./data/resized',
                        help='directory for resized training images')
    parser.add_argument('--caption_path', type=str,
                         default='./data/annotation/results_20130124_train.token',
                         help='path for train annotation file')
    # parser.add_argument('--caption_path', type=str,
    #                    default='./data/annotation/test',
    #                    help='path for train annotation file')

    parser.add_argument('--caption_val_path', type=str,
                         default='./data/annotation/results_20130124_val.token',
                         help='path for validation annotation json file')
    # parser.add_argument('--caption_val_path', type=str,
    #                    default='./data/annotation/test',
    #                    help='path for validation annotation json file')

    parser.add_argument('--det_file', type=str,
                         default='./det/train.hdf5', 
                         help='path for train detection feature hdf5 file')
    parser.add_argument('--det_val_file', type=str,
                         default='./det/val.hdf5',
                         help='path for val detection feature hdf5 file')

    parser.add_argument('--log_step', type=int, default=10,
                        help='step size for printing log info')
    parser.add_argument('--seed', type=int, default=123,
                        help='random seed for model reproduction')

    # ---------------------------Hyper Parameter Setup------------------------------------

    # CNN fine-tuning
    parser.add_argument('--fine_tune_start_layer', type=int, default=5,
                        help='CNN fine-tuning layers from: [0-7]')
    parser.add_argument('--cnn_epoch', type=int, default=20,
                        help='start fine-tuning CNN after')

    # Optimizer Adam parameter
    parser.add_argument('--alpha', type=float, default=0.8,
                        help='alpha in Adam')
    parser.add_argument('--beta', type=float, default=0.999,
                        help='beta in Adam')
    parser.add_argument('--learning_rate', type=float, default=4e-4,
                        help='learning rate for the whole model')
    parser.add_argument('--learning_rate_cnn', type=float, default=1e-4,
                        help='learning rate for fine-tuning CNN')

    # LSTM hyper parameters
    parser.add_argument('--embed_size', type=int, default=256,
                        help='dimension of word embedding vectors, also dimension of v_g')
    parser.add_argument('--hidden_size', type=int, default=512,
                        help='dimension of lstm hidden states')

    # Training details
    parser.add_argument('--pretrained', type=str, default='', help='start from checkpoint or scratch')
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=20)  # on cluster setup, 60 each x 4 for Huckle server
    # parser.add_argument('--batch_size', type=int, default=30)  # on cluster setup, 60 each x 4 for Huckle server

    # For eval_size > 30, it will cause cuda OOM error on Huckleberry
    parser.add_argument('--eval_size', type=int, default=3)  # on cluster setup, 30 each x 4
    # parser.add_argument('--eval_size', type=int, default=14)  # on cluster setup, 30 each x 4
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--clip', type=float, default=0.1)
    parser.add_argument('--lr_decay', type=int, default=20, help='epoch at which to start lr decay')
    parser.add_argument('--learning_rate_decay_every', type=int, default=50,
                        help='decay learning rate at every this number')

    args = parser.parse_args()

    print '------------------------Model and Training Details--------------------------'
    print(args)

    # Start training
    main(args)
    

        
        