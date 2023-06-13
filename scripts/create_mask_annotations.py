import sys
from pathlib import Path
sys.path.append(str(Path(__file__).absolute().parent.parent))
import argparse
import numpy as np
import cv2
from utils import hdri
from utils.constants import *
import re
from PIL import Image
from rembg import remove


def create_images_for_mask_annotation(object_dir: Path):
    """Creates images for mask annotation.
    Args:
        object_dir: Path to an object dir with the source files, e.g., 'source_data/obj'
    """
    tonemap = cv2.createTonemapReinhard()

    env_dirs = sorted([x for x in object_dir.iterdir() if len(x.name) == 1 or x.name in ('train', 'valid', 'test')])

    for env_dir in env_dirs:
        for im_dir in env_dir.iterdir():
            if im_dir.is_dir() and re.match('^test\d$', im_dir.name):
                print(str(im_dir))
                paths = sorted(list(im_dir.glob('*.CR3')))
                p = paths[len(paths)//2:][0]
                raws = [hdri.read_raw_and_meta(p) for p in paths]
                im = hdri.compute_hdri_from_raws(raws)
                im = np.nan_to_num(tonemap.process(im.astype(np.float32)/im.max()))
                im = np.clip(255*im,0,255).astype(np.uint8)

                out_path =env_dir/f'{im_dir.name}.jpg'
                print('writing', str(out_path))
                cv2.imwrite(str(out_path), im[...,[2,1,0]])

                mask_out_path =env_dir/f'{im_dir.name}_mask.png'
                if not mask_out_path.exists():
                    pil_img = Image.fromarray(im)
                    mask = remove(pil_img)
                    mask = np.asarray(mask)[...,-1].copy()
                    mask[mask > 127] = 255
                    mask[mask <= 127] = 0
                    mask = np.stack(3*[np.asarray(mask)], axis=-1)

                    print('writing', str(mask_out_path))
                    cv2.imwrite(str(mask_out_path), mask)


    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Script with helper functions for creating the mask annotations.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("object_dir", type=Path, help="Paths to the source directory that contains images of an object in multiple environments. Environment dirs are named as 'test', 'train', 'valid'")

    if len(sys.argv)==1:
        parser.print_help(sys.stderr)
        sys.exit(1)
    args = parser.parse_args()
    print(args)

    create_images_for_mask_annotation(**vars(args))

