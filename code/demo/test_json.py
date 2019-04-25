import argparse
import json
import os
from os import path as osp
import sys

def change_dir(label_dir, det_path):
    if not osp.exists(label_dir):
        print('Can not find', label_dir)
        return
    print('Processing', label_dir)
    # input_names = [n for n in os.listdir(label_dir)
    #                if osp.splitext(n)[1] == '.json']
    input_names = ['bdd_test_results_15.json']
    boxes = []
    count = 0
    classes = []
    for name in input_names:
        in_path = osp.join(label_dir, name)
        # output_name = name.replace('.json','.txt')
        infos = json.load(open(in_path, 'r'))
        # print(type(out))
        infos_dict = {}
        filenames = []
        for info in infos:
            # print(info['filename'])
            if info['filename'] not in filenames:
                filename = info['filename']
                # file_info = {'filename': filename, 'rects': []}
                filenames.append(filename)
                # filenames.append({"filename": filename, "rects": []})
        # print(filenames)
        print(len(filenames))
        file_infos = []
        print(len(filenames))
        for filename in filenames:
            # print(filename)  
            file_info = []
            for info in infos:
                # print('------------------------')
                # print(filename)
                # print(info['filename'])
                if(filename == info['filename']):
                    # print(filename)
                    # print((info['rects'][0]))
                    # print(info['rects'][0]['confidence'])
                    if(info['rects'][0]['confidence'] > 0.05):
                        file_info.append(info['rects'][0])
            file_dict = {'filename':filename, 'rects': file_info}
            file_infos.append(file_dict)
            # break
        # print((file_dict))
        # print(file_infos)
    image_infos = {'results': file_infos}    
    fw = open(det_path, 'w')
    json.dump(image_infos, fw)#, indent=4, separators=(',', ': '))
        # for info in infos:
        #     # print(info['filename'])
        #     if info['filename'] not in filenames:
        #         filename = info['filename']
        #         # file_info = {'filename': filename, 'rects': []}
        #         filenames.append(filename)

     

def main():
    # args = parse_args()
    # change_dir(args.label_dir, args.det_path)
    change_dir('./','./test_result_15.json')
if __name__ == '__main__':
    main()
