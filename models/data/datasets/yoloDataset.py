import os
import glob
import tqdm
import torch
import numpy as np
from torch.utils.data.dataset import Dataset
from pathlib import Path
from PIL import Image


img_formats = ['bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng', 'webp', 'mpo']


class YOLODataset(Dataset):

    def __init__(self, data_dir, img_size, ):
        super().__init__()
        self.data_dir = data_dir
        self.img_size = img_size

        image_dir = os.path.join(data_dir, "images")

        f = []  # image files
        f += glob.glob(str(image_dir / '**' / '*.*'), recursive=True)
        for x in f:
            if x.split('.')[-1].lower() in img_formats:
                x.replace('/', os.sep)
            else:
                raise Exception(f'{x} is not a processable image.')
        self.img_files = sorted(f)
        assert self.img_files, 'No images found.'

        self.label_files = img2label_path(self.img_files)
        cache_path = Path(self.label_files[0]).parent.with_suffix('.cache')  # cached labels
        if cache_path.is_file():
            cache, exists = torch.load(cache_path), True  # load
            if cache['hash'] != get_hash(self.label_files + self.img_files) or 'version' not in cache:  # changed
                cache, exists = self.cache_labels(cache_path), False  # re-cache
        else:
            cache, exists = self.cache_labels(cache_path), False  # cache
        
        nf, nm, ne, n = cache.pop('results')  # found, missing, empty, total
        if exists:
            print(f"Scanning '{cache_path}' images and labels... {nf} found, {nm} missing, {ne} empty")
        assert nf > 0, f'No labels in {cache_path}. Can not train without labels.'

        # Read cache
        cache.pop('hash')  # remove hash
        cache.pop('version')  # remove version
        labels, shapes = zip(*cache.values())
        self.labels = list(labels)
        self.shapes = np.array(shapes, dtype=np.float64)
        self.img_files = list(cache.keys())  # update
        self.label_files = img2label_paths(cache.keys())  # update
        if single_cls:
            for x in self.labels:
                x[:, 0] = 0

        n = len(shapes)
        bi = np.floor(np.arange(n) / batch_size).astype(np.int)  # batch index
        nb = bi[-1] + 1


    def cache_labels(self, path=Path('./labels.cache')):
        # Cache dataset labels, check images and read shapes
        x = {}  # dict
        nm, nf, ne, nc = 0, 0, 0, 0  # number missing, found, empty, duplicate
        pbar = tqdm(zip(self.img_files, self.label_files), desc='Scanning images', total=len(self.img_files))
        for i, (im_file, lb_file) in enumerate(pbar):
            # verify images
            im = Image.open(im_file)
            im.verify()  # PIL verify
            shape = im.size

            # verify labels
            if os.path.isfile(lb_file):
                nf += 1  # label found
                with open(lb_file, 'r') as f:
                    l = [x.split() for x in f.read().strip().splitlines()]
                    l = np.array(l, dtype=np.float32)
                if len(l):
                    assert l.shape[1] == 5, 'labels require 5 columns each'
                    assert (l >= 0).all(), 'negative labels'
                    assert (l[:, 1:] <= 1).all(), 'non-normalized or out of bounds coordinate labels'
                    assert np.unique(l, axis=0).shape[0] == l.shape[0], 'duplicate labels'
                else:
                    ne += 1  # label empty
                    l = np.zeros((0, 5), dtype=np.float32)
            else:
                nm += 1  # label missing
                l = np.zeros((0, 5), dtype=np.float32)
            x[im_file] = [l, shape]
            pbar.desc = f"Scanning '{path.parent / path.stem}' images and labels... " \
                        f"{nf} found, {nm} missing, {ne} empty"
        pbar.close()

        if nf == 0:
            print(f'WARNING: No labels found in {path}.')

        x['hash'] = get_hash(self.label_files + self.img_files)
        x['results'] = nf, nm, ne, i + 1
        x['version'] = 1.0  # when the cache code changed, change the version.
        torch.save(x, path)
        return x


def get_hash(files):
    # Returns a single hash value of a list of files
    return sum(os.path.getsize(f) for f in files if os.path.isfile(f))


def img2label_paths(img_paths):
    # Define label paths as a function of image paths
    sa, sb = os.sep + 'images' + os.sep, os.sep + 'labels' + os.sep  # /images/, /labels/ substrings
    return ['txt'.join(x.replace(sa, sb, 1).rsplit(x.split('.')[-1], 1)) for x in img_paths]
