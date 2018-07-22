
"""
Copyright (c) 2018. All rights reserved.
Created by Resnick Xing on 2018/5/11
"""

from perception.infers.segmention_infer import SegmentionInfer
from perception.metric.segmention_metric import *
from configs.utils.config_utils import process_config


repredict=True

def main_test():
    print('[INFO] Reading Configs...')
    config = None

    try:
        config = process_config('configs/segmention_config.json')
    except Exception as e:
        print('[Exception] Config Error, %s' % e)
        exit(0)

    if repredict==True:

        print('[INFO] Predicting...')
        infer = SegmentionInfer( config)
        infer.predict()

    print('[INFO] Metric results...')
    gtlist=fileList(config.test_gt_path,'*'+config.test_gt_datatype)
    problist=fileList(config.test_result_path,'*.bmp')
    modelName=['DenseNet-Unet']
    drawCurve(gtlist,[problist],modelName,'DRIVE',config.checkpoint)

    print('[INFO] Fininshed...')


if __name__ == '__main__':
    main_test()
