# -*- coding: utf-8 -*-
# @Time : 2023/4/24 16:27
# @Author : mr.felix
# @Email ： 2578925789@qq.com
# @File : pycocotools_test
# @Description : 读取检测结果保存的json调用pycocotools计算AP并保存npy

import json
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

names = ['face', 'hand', 'cigarette', 'cellphone']

def comput_mAP(json_pr, json_gt, save_dir):
    coco_gt = COCO(json_gt)

    coco_dt = coco_gt.loadRes(json_pr)

    cocoEval = COCOeval(coco_gt, coco_dt, 'bbox')
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()
    cocoEval.PR_Curve(names, json_gt, save_dir, max_indx=300)  # max_indx=300对应conf=0.3时的P、R取值

if __name__ == "__main__":
    pred_json = "./save/detect_reslut.json"
    val_json = './save/val.json'
    save_dir = "./save/PR_result/"
    with open(pred_json, "r", encoding="utf-8") as f:
        json_content = json.load(f)
        dt_num = len(json_content)
        print("dt_num:{}".format(dt_num))
    comput_mAP(pred_json, val_json, save_dir)
