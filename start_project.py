import os
import argparse
import glob
import shutil


if __name__ == '__main__':
    # argments settings
    parser = argparse.ArgumentParser()
    parser.add_argument('pj_name')
    args = parser.parse_args()

    # Make new project directory
    root = f'./{args.pj_name}'
    sub_dir1 = ['original_data', 'processed_data', 'results', 'src']
    sub_dir2 = [
        ['best_params', 'trained_models', 'valid_metrics'],
        ['pipeline', 'descriptors']
        ]
    for dir1 in sub_dir1:
        dir = os.path.join(root, dir1)
        if dir1 == 'results':
            for dir2 in sub_dir2[0]:
                dir_ = os.path.join(dir, dir2)
                if os.path.exists(dir_) == False:
                    os.makedirs(dir_)
        elif dir1 == 'src':
            for dir2 in sub_dir2[1]:
                dir_ = os.path.join(dir, dir2)
                if os.path.exists(dir_) == False:
                    os.makedirs(dir_)
        else:
            if os.path.exists(dir) == False:
                os.makedirs(dir)
    
    # copy python files
    root_source = './src'
    root_copy = f'./{args.pj_name}/src'
    for path in sub_dir2[1]:
        path_ = os.path.join(root_source, path)
        path_ = os.path.join(path_, '*.py')
        path_sources = glob.glob(path_)
        for path_source in path_sources:
            path_copy = os.path.join(root_copy, path)
            path_copy = os.path.join(path_copy, path_source.split('/')[-1])
            shutil.copyfile(path_source,path_copy)
    
    # copy setting file
    path_source = os.path.join(root_source, 'settings.ini')
    path_copy = os.path.join(root_copy, 'settings.ini')
    shutil.copyfile(path_source,path_copy)