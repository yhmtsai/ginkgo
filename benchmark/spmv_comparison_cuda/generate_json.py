import numpy as np
import h5py
from scipy import io
import json
import sys
import os

# Usage:
# Matlab: use combine_spmv_result to combine results and get the mat file
#       [out_legend, out_data] = combine_spmv_result('k40m_benchmark', '_n1')
#       save('k40m.mat', 'out_legend', 'out_data')
# Note: the variable 'out_legend' and 'out_data' must be correct.
# Python: use generate_json to convert mat to JSON for GPE
#       python generate_json <info_dir> <mat> <mtx_list> <output_dir>
#       python generate_json /path/to/K20Xm/cuda/SuiteSparse k40m.mat realmtx.list /path/to/K40m/cuda/SuiteSparse
# Note: if reading json file is failed, the problem info only contain 'group', 'name', 'rows', 'cols', 'nonzeros', 'real'
#       'nonzero' is token from mtx_reader. it may be more like the number of stored elements.

def load_mat(matfile):
    # catch out_legend and out_data
    try:
        mat = io.loadmat(matfile)
        out_legend = [element[0] for element in mat['out_legend'][0]]
        out_data = mat['out_data']
    except:
        # For matlab v7.3 mat
        mat = h5py.File(matfile)
        out_legend = [''.join(map(unichr, mat[element[0]][:])) for element in mat['out_legend']]
        out_data = np.transpose(mat['out_data'])
    return out_legend, out_data

def build_mtx_list(mtx_list):
    with open(mtx_list) as f:
        lines = [line.rstrip('\n') for line in f]
    return lines

def get_mtx_info(mtx):
    group = os.path.dirname(mtx)
    basename = os.path.basename(mtx)
    name, _ = os.path.splitext(basename)
    return group, name

if __name__ == "__main__":
    if len(sys.argv)!= 5:
        print("Usage:" + sys.argv[0] + " <info_dir> <mat> <mtx_list> <output_dir>")
        sys.exit(1)
    
    INFO_DIR = sys.argv[1]
    MAT = sys.argv[2]
    MTX_LIST = sys.argv[3]
    OUTPUT_DIR = sys.argv[4]
    # Create OUTPUT_DIR
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    legend, results = load_mat(MAT)
    mtx_list = build_mtx_list(MTX_LIST)
    for i in range(len(mtx_list)):
        mtx = mtx_list[i]
        line = results[i, :]
        group, name = get_mtx_info(mtx)
        json_name = group + '/' + name + '.json'
        # load INFO_DIR/group/name.json
        try:
            with open(INFO_DIR + '/' + json_name, 'r') as f:
                data = json.load(f)
        except:
            data[0]["filename"] = "/home/benchmarks/.config/ssget/MM/" + group + '/' + name + '.mtx'
            data[0]["problem"] = {
                "group": group,
                "name": name,
                "rows": line[0],
                "cols": line[1],
                "nonzeros": line[2],
                "real": True,
            }

        # overwrite all data into spmv
        temp_data = dict()
        best_format = 'none'
        best_time = np.Inf
        for j in range(len(legend)):
            us = line[2*j+4]
            if us < 0:
                temp_data[legend[j]] = {'storage': 0, 'time': 0, 'completed': False}
            else:
                ns = us * 1e3
                temp_data[legend[j]] = {'storage': 0, 'time': ns, 'completed': True}
                if ns < best_time:
                    best_time = ns
                    best_format = legend[j]
        data[0]['spmv'] = temp_data
        data[0]['optimal']['spmv'] = best_format
        # output in OUTPUT_DIR/group/name.json
        if not os.path.exists(OUTPUT_DIR + '/' + group):
            os.makedirs(OUTPUT_DIR + '/' + group)
        with open(OUTPUT_DIR + '/' + json_name , 'w') as f:
            json.dump(data, f, indent=4)


