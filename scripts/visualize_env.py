import sys
from pathlib import Path
sys.path.append(str(Path(__file__).absolute().parent.parent))
import re
import numpy as np
import pycolmap
import argparse
from utils import reconstruction as sfm
from utils import equirectangular
from utils import hdri
from utils.constants import INTERMEDIATE_DATA_PATH


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Script that visualizes an environment from the dataset with the RPC interface.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("env_dir", type=Path, help="Paths to the environment directory of the final dataset.")

    if len(sys.argv)==1:
        parser.print_help(sys.stderr)
        sys.exit(1)
    args = parser.parse_args()
    print(args)

    env_dir = args.env_dir.resolve()
    env_name = env_dir.name
    obj_name = env_dir.parent.name

    recon_dir = INTERMEDIATE_DATA_PATH/obj_name/env_name/'recon'
    db_path = recon_dir/'database.db'

    recon = pycolmap.Reconstruction(recon_dir/'0')
    custom_keypoints = sfm.get_custom_keypoints_from_db(db_path)
    sfm.visualize_reconstruction(recon, custom_keypoints, sg_prefix=f'{recon_dir.parent.parent.name}.{recon_dir.parent.name}')



    envmap_paths = sorted(list(env_dir.glob('*.hdr')))

    for envmap_p in envmap_paths:
        transform_path = envmap_p.parent/envmap_p.name.replace('env', 'world_to_env').replace('.hdr', '.txt')
        if transform_path.exists():
            print(str(envmap_p), str(transform_path))
            w2env = np.loadtxt(transform_path)
        else:
            print(str(envmap_p), 'no transform found')
            w2env = np.eye(4)
        hdr = hdri.read_hdri(envmap_p)
        ldr = hdri.compute_ldr_from_hdri_opencv(hdr)
        equirectangular.visualize_envmap_as_sphere(ldr, w2env, prefix=f'{obj_name}.{env_name}/{envmap_p.stem}')
