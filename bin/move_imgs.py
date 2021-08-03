#!/opt/conda/envs/tensorEnv/bin/python

import argparse
import os
import shutil

def move_to_directory(Dir):
    srcs = os.listdir(Dir)
    srcs_t = list(filter(lambda x: x.startswith('human_true'), srcs))
    srcs_f = list(filter(lambda x: x.startswith('human_false'), srcs))
    
    dest = os.path.join(Dir, 'human-true')
    def move_imgs(img):
        shutil.copy(img, dest)
    
    list(map(move_imgs, srcs_t))

    dest = os.path.join(Dir, 'human-false')
    list(map(move_imgs, srcs_f))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='Move cgr images to train and test directories',)
    
    parser.add_argument('--source', default=None,
                        help='Directory containing CGR images')
    args = parser.parse_args()

    move_to_directory(args.source)