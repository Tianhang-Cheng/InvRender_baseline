import sys
from pathlib import Path
sys.path.append(str(Path(__file__).absolute().parent.parent))
print(sys.path)
import numpy as np
import argparse
import json
from utils.metrics import METRICS
from utils.tabelle import *


def create_tables(methods_data, subset='test'):
    num_methods = len(methods_data)

    object_names = set()
    for x in methods_data:
        for gt_image_path in x['data'].keys():
            subset_name = Path(gt_image_path).parent.name
            obj_name = Path(gt_image_path).parent.parent.name
            object_names.add(obj_name) 


    all_gt_image_names = ['gt_image_{:04d}.png'.format(x) for x in range(9)]
    if subset == 'test':
        same_env_gt_image_names = all_gt_image_names[:3]
        new_env_gt_image_names = all_gt_image_names[3:]
    else:
        raise Exception("only the 'test' subset is supported")

    num_col_blocks = 3 # we have three column blocks for 'same env', 'new env', 'mean'

    table = Table((num_methods+2,1+num_col_blocks*len(METRICS)))

    table[0,0].rowfmt.topmost_line = True
    table[0,0].colfmt.align = 'l'
    table[1,0] = Cell('Method', bold=True)
    table[0,0].rowfmt.line = [(1,3), (4,6), (7,9)]
    table[1,0].rowfmt.line = True
    table[0,1] = Cell('Same environment', col_span=3, bold=True)
    table[0,4] = Cell('New environment', col_span=3, bold=True) 
    table[0,7] = Cell('Mean', col_span=3, bold=True)
    table[1,1:] = num_col_blocks*[x['latex'] for _, x in METRICS.items()]
    table[-1,0].rowfmt.line = True
    for i in range(num_col_blocks):
        for metric_i, (metric_name, metric) in enumerate(METRICS.items()):
            table[0,num_col_blocks*i+metric_i+1].colfmt.auto_highlight = metric['best']
            table[0,num_col_blocks*i+metric_i+1].colfmt.num_format = '{:2.3f}' if metric_name == 'LPIPS' else '{:2.2f}'


    for method_i, md in enumerate(methods_data):
        row = method_i+2
        table[method_i+2,0] = md['name']

        for env_i, gt_image_names in enumerate((same_env_gt_image_names, new_env_gt_image_names, all_gt_image_names)):
            for metric_i, metric in enumerate(METRICS):
                values = []
                for obj in object_names:
                    for gt_im in gt_image_names:  
                        if f'{obj}/{subset}/{gt_im}' in md['data'].keys():
                            values.append(md['data'][f'{obj}/{subset}/{gt_im}'].get(metric, np.nan))
                col = len(METRICS)*env_i+metric_i+1
                # table[row,col] = np.nanmean(values)
                table[row,col] = np.mean(values)

    print('subset', subset)
    print(table)

    print('\n--- Latex\n')
    latex_str = table.latex()
    print(latex_str)
    print('\n---\n')
    return latex_str


if __name__ == '__main__':
    import glob
    parser = argparse.ArgumentParser(
        description="Script that creates latex tables from the json files generated by the evaluate.py script.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    # parser.add_argument('inputs', type=Path, nargs='+', help="Path to the input json files")
    parser.add_argument('--output', type=Path, help="Path to the output tex file")
    # parser.add_argument('--set', choices=set(['train', 'valid', 'test']), nargs="+", default=['test'], help="The subset. This is usually 'test'")

    if len(sys.argv)==1:
        parser.print_help(sys.stderr)
        sys.exit(1)
    args = parser.parse_args()
    print(args)

    inputs = glob.glob(' relits/*.json')
    inputs = sorted(inputs)

    methods_data = [] 
    for p in inputs:
        with open(p,'r') as f:
            methods_data.append(json.load(f))

    for subset in ('test',):
        latex = create_tables(methods_data=methods_data, subset=subset)
        if args.output:
            with open(args.output, 'w') as f:
                f.write(latex)