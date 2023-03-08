from data import COCODetection ,get_label_map ,MEANS ,COLORS #line:1
from yolact import Yolact #line:2
from utils .augmentations import BaseTransform ,FastBaseTransform ,Resize #line:3
from utils .functions import MovingAverage ,ProgressBar #line:4
from layers .box_utils import jaccard ,center_size ,mask_iou #line:5
from utils import timer #line:6
from utils .functions import SavePath #line:7
from layers .output_utils import postprocess ,undo_image_transformation #line:8
import pycocotools #line:9
from data import cfg ,set_cfg ,set_dataset #line:10
import numpy as np #line:11
import torch #line:12
import torch .backends .cudnn as cudnn #line:13
from torch .autograd import Variable #line:14
import argparse #line:15
import time #line:16
import random #line:17
import cProfile #line:18
import pickle #line:19
import json #line:20
import os #line:21
from collections import defaultdict #line:22
from pathlib import Path #line:23
from collections import OrderedDict #line:24
from PIL import Image #line:25
import matplotlib .pyplot as plt #line:26
import cv2 #line:27
import datetime #line:28
def str2bool (OOO00000OO00OOOOO ):#line:31
    if OOO00000OO00OOOOO .lower ()in ('yes','true','t','y','1'):#line:32
        return True #line:33
    elif OOO00000OO00OOOOO .lower ()in ('no','false','f','n','0'):#line:34
        return False #line:35
    else :#line:36
        raise argparse .ArgumentTypeError ('Boolean value expected.')#line:37
def parse_args (argv =None ):#line:40
    OO00OO000O0O00O00 =argparse .ArgumentParser (description ='YOLACT COCO Evaluation')#line:42
    OO00OO000O0O00O00 .add_argument ('--trained_model',default ='./weights/yolact_plus_resnet50_slab_98_38000.pth',type =str ,help ='Trained state_dict file path to open. If "interrupt", this will open the interrupt file.')#line:45
    OO00OO000O0O00O00 .add_argument ('--top_k',default =1 ,type =int ,help ='Further restrict the number of predictions to parse')#line:47
    OO00OO000O0O00O00 .add_argument ('--cuda',default =True ,type =str2bool ,help ='Use cuda to evaulate model')#line:49
    OO00OO000O0O00O00 .add_argument ('--fast_nms',default =True ,type =str2bool ,help ='Whether to use a faster, but not entirely correct version of NMS.')#line:51
    OO00OO000O0O00O00 .add_argument ('--cross_class_nms',default =False ,type =str2bool ,help ='Whether compute NMS cross-class or per-class.')#line:53
    OO00OO000O0O00O00 .add_argument ('--display_masks',default =True ,type =str2bool ,help ='Whether or not to display masks over bounding boxes')#line:55
    OO00OO000O0O00O00 .add_argument ('--display_bboxes',default =True ,type =str2bool ,help ='Whether or not to display bboxes around masks')#line:57
    OO00OO000O0O00O00 .add_argument ('--display_text',default =True ,type =str2bool ,help ='Whether or not to display text (class [score])')#line:59
    OO00OO000O0O00O00 .add_argument ('--display_scores',default =True ,type =str2bool ,help ='Whether or not to display scores in addition to classes')#line:61
    OO00OO000O0O00O00 .add_argument ('--display',dest ='display',action ='store_true',help ='Display qualitative results instead of quantitative ones.')#line:63
    OO00OO000O0O00O00 .add_argument ('--shuffle',dest ='shuffle',action ='store_true',help ='Shuffles the images when displaying them. Doesn\'t have much of an effect when display is off though.')#line:65
    OO00OO000O0O00O00 .add_argument ('--ap_data_file',default ='results/ap_data.pkl',type =str ,help ='In quantitative mode, the file to save detections before calculating mAP.')#line:67
    OO00OO000O0O00O00 .add_argument ('--resume',dest ='resume',action ='store_true',help ='If display not set, this resumes mAP calculations from the ap_data_file.')#line:69
    OO00OO000O0O00O00 .add_argument ('--max_images',default =-1 ,type =int ,help ='The maximum number of images from the dataset to consider. Use -1 for all.')#line:71
    OO00OO000O0O00O00 .add_argument ('--output_coco_json',dest ='output_coco_json',action ='store_true',help ='If display is not set, instead of processing IoU values, this just dumps detections into the coco json file.')#line:73
    OO00OO000O0O00O00 .add_argument ('--bbox_det_file',default ='results/bbox_detections.json',type =str ,help ='The output file for coco bbox results if --coco_results is set.')#line:75
    OO00OO000O0O00O00 .add_argument ('--mask_det_file',default ='results/mask_detections.json',type =str ,help ='The output file for coco mask results if --coco_results is set.')#line:77
    OO00OO000O0O00O00 .add_argument ('--config',default ='yolact_resnet50_slab_config',help ='The config object to use.')#line:79
    OO00OO000O0O00O00 .add_argument ('--output_web_json',dest ='output_web_json',action ='store_true',help ='If display is not set, instead of processing IoU values, this dumps detections for usage with the detections viewer web thingy.')#line:81
    OO00OO000O0O00O00 .add_argument ('--web_det_path',default ='web/dets/',type =str ,help ='If output_web_json is set, this is the path to dump detections into.')#line:83
    OO00OO000O0O00O00 .add_argument ('--no_bar',dest ='no_bar',action ='store_true',help ='Do not output the status bar. This is useful for when piping to a file.')#line:85
    OO00OO000O0O00O00 .add_argument ('--display_lincomb',default =False ,type =str2bool ,help ='If the config uses lincomb masks, output a visualization of how those masks are created.')#line:87
    OO00OO000O0O00O00 .add_argument ('--benchmark',default =False ,dest ='benchmark',action ='store_true',help ='Equivalent to running display mode but without displaying an image.')#line:89
    OO00OO000O0O00O00 .add_argument ('--no_sort',default =False ,dest ='no_sort',action ='store_true',help ='Do not sort images by hashed image ID.')#line:91
    OO00OO000O0O00O00 .add_argument ('--seed',default =None ,type =int ,help ='The seed to pass into random.seed. Note: this is only really for the shuffle and does not (I think) affect cuda stuff.')#line:93
    OO00OO000O0O00O00 .add_argument ('--mask_proto_debug',default =False ,dest ='mask_proto_debug',action ='store_true',help ='Outputs stuff for scripts/compute_mask.py.')#line:95
    OO00OO000O0O00O00 .add_argument ('--no_crop',default =False ,dest ='crop',action ='store_false',help ='Do not crop output masks with the predicted bounding box.')#line:97
    OO00OO000O0O00O00 .add_argument ('--image',default =None ,type =str ,help ='A path to an image to use for display.')#line:99
    OO00OO000O0O00O00 .add_argument ('--images',default ='./data/slab/real_test_20211116:output_images',type =str ,help ='An input folder of images and output folder to save detected images. Should be in the format input->output.')#line:103
    OO00OO000O0O00O00 .add_argument ('--video',default =None ,type =str ,help ='A path to a video to evaluate on. Passing in a number will use that index webcam.')#line:107
    OO00OO000O0O00O00 .add_argument ('--video_multiframe',default =1 ,type =int ,help ='The number of frames to evaluate in parallel to make videos play at higher fps.')#line:109
    OO00OO000O0O00O00 .add_argument ('--score_threshold',default =0.10 ,type =float ,help ='Detections with a score under this threshold will not be considered. This currently only works in display mode.')#line:111
    OO00OO000O0O00O00 .add_argument ('--dataset',default =None ,type =str ,help ='If specified, override the dataset specified in the config with this one (example: coco2017_dataset).')#line:113
    OO00OO000O0O00O00 .add_argument ('--detect',default =False ,dest ='detect',action ='store_true',help ='Don\'t evauluate the mask branch at all and only do object detection. This only works for --display and --benchmark.')#line:115
    OO00OO000O0O00O00 .add_argument ('--display_fps',default =False ,dest ='display_fps',action ='store_true',help ='When displaying / saving video, draw the FPS on the frame')#line:117
    OO00OO000O0O00O00 .add_argument ('--emulate_playback',default =False ,dest ='emulate_playback',action ='store_true',help ='When saving a video, emulate the framerate that you\'d get running in real-time mode.')#line:119
    OO00OO000O0O00O00 .set_defaults (no_bar =False ,display =False ,resume =False ,output_coco_json =False ,output_web_json =False ,shuffle =False ,benchmark =False ,no_sort =False ,no_hash =False ,mask_proto_debug =False ,crop =True ,detect =False ,display_fps =False ,emulate_playback =False )#line:125
    global args #line:127
    args =OO00OO000O0O00O00 .parse_args (argv )#line:128
    if args .output_web_json :#line:130
        args .output_coco_json =True #line:131
    if args .seed is not None :#line:133
        random .seed (args .seed )#line:134
iou_thresholds =[OOO0OOOOO000OO0OO /100 for OOO0OOOOO000OO0OO in range (50 ,100 ,5 )]#line:137
coco_cats ={}#line:138
coco_cats_inv ={}#line:139
color_cache =defaultdict (lambda :{})#line:140
def prep_display (O0O0OOO0OO0O0OOOO ,O0OO00OOOOOOOO0O0 ,OOOO000OO000000OO ,OO0O00O0OOO0O00O0 ,undo_transform =True ,class_color =False ,mask_alpha =0.45 ,fps_str =''):#line:143
    ""#line:146
    if undo_transform :#line:147
        OOOO00OO0000O00OO =undo_image_transformation (O0OO00OOOOOOOO0O0 ,OO0O00O0OOO0O00O0 ,OOOO000OO000000OO )#line:148
        O0OO00OO00OO00O00 =torch .Tensor (OOOO00OO0000O00OO ).cuda ()#line:149
    else :#line:150
        O0OO00OO00OO00O00 =O0OO00OOOOOOOO0O0 /255.0 #line:151
        OOOO000OO000000OO ,OO0O00O0OOO0O00O0 ,_O0OOO0O000O00OOO0 =O0OO00OOOOOOOO0O0 .shape #line:152
    with timer .env ('Postprocess'):#line:154
        OO00O0000OOO0OO00 =cfg .rescore_bbox #line:155
        cfg .rescore_bbox =True #line:156
        OOOOOO00OOO00OO0O =postprocess (O0O0OOO0OO0O0OOOO ,OO0O00O0OOO0O00O0 ,OOOO000OO000000OO ,visualize_lincomb =args .display_lincomb ,crop_masks =args .crop ,score_threshold =args .score_threshold )#line:159
        cfg .rescore_bbox =OO00O0000OOO0OO00 #line:160
    with timer .env ('Copy'):#line:162
        O0O0O00OOO0O000O0 =OOOOOO00OOO00OO0O [1 ].argsort (0 ,descending =True )[:args .top_k ]#line:163
        if cfg .eval_mask_branch :#line:165
            O0OO00O0O0OO00OOO =OOOOOO00OOO00OO0O [3 ][O0O0O00OOO0O000O0 ]#line:167
        OOO000OO0OO0O0O00 ,O0O00O0OOO000OOO0 ,O0OO000O0OO00OO0O =[O00OO0OOO00O0O0OO [O0O0O00OOO0O000O0 ].cpu ().numpy ()for O00OO0OOO00O0O0OO in OOOOOO00OOO00OO0O [:3 ]]#line:168
    OOO00O0000O000OOO =min (args .top_k ,OOO000OO0OO0O0O00 .shape [0 ])#line:170
    for O0OO00OO0000OOOOO in range (OOO00O0000O000OOO ):#line:171
        if O0O00O0OOO000OOO0 [O0OO00OO0000OOOOO ]<args .score_threshold :#line:172
            OOO00O0000O000OOO =O0OO00OO0000OOOOO #line:173
            break #line:174
    def O0000OOOOOOO00O0O (O000O0OO0000OOO00 ,on_gpu =None ):#line:178
        global color_cache #line:179
        O00OO0OO0OOO0OO00 =(OOO000OO0OO0O0O00 [O000O0OO0000OOO00 ]*5 if class_color else O000O0OO0000OOO00 *5 )%len (COLORS )#line:180
        if on_gpu is not None and O00OO0OO0OOO0OO00 in color_cache [on_gpu ]:#line:182
            return color_cache [on_gpu ][O00OO0OO0OOO0OO00 ]#line:183
        else :#line:184
            OOO00OOO0OO0OOO0O =COLORS [O00OO0OO0OOO0OO00 ]#line:185
            if not undo_transform :#line:186
                OOO00OOO0OO0OOO0O =(OOO00OOO0OO0OOO0O [2 ],OOO00OOO0OO0OOO0O [1 ],OOO00OOO0OO0OOO0O [0 ])#line:188
            if on_gpu is not None :#line:189
                OOO00OOO0OO0OOO0O =torch .Tensor (OOO00OOO0OO0OOO0O ).to (on_gpu ).float ()/255. #line:190
                color_cache [on_gpu ][O00OO0OO0OOO0OO00 ]=OOO00OOO0OO0OOO0O #line:191
            return OOO00OOO0OO0OOO0O #line:192
    if args .display_masks and cfg .eval_mask_branch and OOO00O0000O000OOO >0 :#line:197
        O0OO00O0O0OO00OOO =O0OO00O0O0OO00OOO [:OOO00O0000O000OOO ,:,:,None ]#line:199
        O000O0O00OOO0O0OO =torch .cat ([O0000OOOOOOO00O0O (O00O00000O0OO0000 ,on_gpu =O0OO00OO00OO00O00 .device .index ).view (1 ,1 ,1 ,3 )for O00O00000O0OO0000 in range (OOO00O0000O000OOO )],dim =0 )#line:203
        O0000OOOOO0O00O0O =O0OO00O0O0OO00OOO .repeat (1 ,1 ,1 ,3 )*O000O0O00OOO0O0OO *mask_alpha #line:204
        O0O000O000O00OOOO =O0OO00O0O0OO00OOO *(-mask_alpha )+1 #line:207
        O0OO0OO000O00OO00 =O0000OOOOO0O00O0O [0 ]#line:212
        if OOO00O0000O000OOO >1 :#line:213
            O00OOOO0000O0O00O =O0O000O000O00OOOO [:(OOO00O0000O000OOO -1 )].cumprod (dim =0 )#line:214
            O0O0000OOO0OOO0O0 =O0000OOOOO0O00O0O [1 :]*O00OOOO0000O0O00O #line:215
            O0OO0OO000O00OO00 +=O0O0000OOO0OOO0O0 .sum (dim =0 )#line:216
        O0OO00OO00OO00O00 =O0OO00OO00OO00O00 *O0O000O000O00OOOO .prod (dim =0 )+O0OO0OO000O00OO00 #line:218
    if args .display_fps :#line:220
        OOO00000OOOO0OO00 =cv2 .FONT_HERSHEY_DUPLEX #line:222
        O0O0OO000O0O000O0 =0.6 #line:223
        O00OOO0OO0O00OO0O =1 #line:224
        OO0OOO000O0OO00O0 ,OOOO0OOOOO00OOO0O =cv2 .getTextSize (fps_str ,OOO00000OOOO0OO00 ,O0O0OO000O0O000O0 ,O00OOO0OO0O00OO0O )[0 ]#line:226
        O0OO00OO00OO00O00 [0 :OOOO0OOOOO00OOO0O +8 ,0 :OO0OOO000O0OO00O0 +8 ]*=0.6 #line:228
    OOOO00OO0000O00OO =(O0OO00OO00OO00O00 *255 ).byte ().cpu ().numpy ()#line:232
    if args .display_fps :#line:234
        OO0O00O00O0OO0O00 =(4 ,OOOO0OOOOO00OOO0O +2 )#line:236
        O0OOO00OO000OO0O0 =[255 ,255 ,255 ]#line:237
        cv2 .putText (OOOO00OO0000O00OO ,fps_str ,OO0O00O00O0OO0O00 ,OOO00000OOOO0OO00 ,O0O0OO000O0O000O0 ,O0OOO00OO000OO0O0 ,O00OOO0OO0O00OO0O ,cv2 .LINE_AA )#line:239
    if OOO00O0000O000OOO ==0 :#line:241
        return OOOO00OO0000O00OO #line:242
    if args .display_text or args .display_bboxes :#line:244
        for O0OO00OO0000OOOOO in reversed (range (OOO00O0000O000OOO )):#line:245
            OOO0OOOOOO0OOO0OO ,O000O0O0O0OOO0000 ,OOOOOOOO0OO00O0OO ,OOO00O0000OOO0OOO =O0OO000O0OO00OO0O [O0OO00OO0000OOOOO ,:]#line:246
            OOO0O00OO0O0000O0 =O0000OOOOOOO00O0O (O0OO00OO0000OOOOO )#line:247
            O0OOO000OOO0OOOOO =O0O00O0OOO000OOO0 [O0OO00OO0000OOOOO ]#line:248
            global g_x1 #line:250
            g_x1 =OOO0OOOOOO0OOO0OO #line:251
            global g_x2 #line:252
            g_x2 =OOOOOOOO0OO00O0OO #line:253
            global g_y1 #line:254
            g_y1 =O000O0O0O0OOO0000 #line:255
            global g_y2 #line:256
            g_y2 =OOO00O0000OOO0OOO #line:257
            if args .display_bboxes :#line:260
                cv2 .rectangle (OOOO00OO0000O00OO ,(OOO0OOOOOO0OOO0OO ,O000O0O0O0OOO0000 ),(OOOOOOOO0OO00O0OO ,OOO00O0000OOO0OOO ),OOO0O00OO0O0000O0 ,1 )#line:261
            if args .display_text :#line:263
                _OOO00OOO0O0OOOOO0 =cfg .dataset .class_names [OOO000OO0OO0O0O00 [O0OO00OO0000OOOOO ]]#line:264
                OOO0OO0O00O0O0O0O ='%s: %.2f'%(_OOO00OOO0O0OOOOO0 ,O0OOO000OOO0OOOOO )if args .display_scores else _OOO00OOO0O0OOOOO0 #line:265
                OOO00000OOOO0OO00 =cv2 .FONT_HERSHEY_DUPLEX #line:267
                O0O0OO000O0O000O0 =0.6 #line:268
                O00OOO0OO0O00OO0O =1 #line:269
                OO0OOO000O0OO00O0 ,OOOO0OOOOO00OOO0O =cv2 .getTextSize (OOO0OO0O00O0O0O0O ,OOO00000OOOO0OO00 ,O0O0OO000O0O000O0 ,O00OOO0OO0O00OO0O )[0 ]#line:271
                OO0O00O00O0OO0O00 =(OOO0OOOOOO0OOO0OO ,O000O0O0O0OOO0000 -3 )#line:273
                O0OOO00OO000OO0O0 =[255 ,255 ,255 ]#line:274
                cv2 .rectangle (OOOO00OO0000O00OO ,(OOO0OOOOOO0OOO0OO ,O000O0O0O0OOO0000 ),(OOO0OOOOOO0OOO0OO +OO0OOO000O0OO00O0 ,O000O0O0O0OOO0000 -OOOO0OOOOO00OOO0O -4 ),OOO0O00OO0O0000O0 ,-1 )#line:276
                cv2 .putText (OOOO00OO0000O00OO ,OOO0OO0O00O0O0O0O ,OO0O00O00O0OO0O00 ,OOO00000OOOO0OO00 ,O0O0OO000O0O000O0 ,O0OOO00OO000OO0O0 ,O00OOO0OO0O00OO0O ,cv2 .LINE_AA )#line:278
    return OOOO00OO0000O00OO #line:280
def prep_benchmark (O00OO0O000000O0OO ,OO0OOOO0O000O000O ,O0OOOOO0OOOO00OOO ):#line:283
    with timer .env ('Postprocess'):#line:284
        OO0OOOO0O0OOO0000 =postprocess (O00OO0O000000O0OO ,O0OOOOO0OOOO00OOO ,OO0OOOO0O000O000O ,crop_masks =args .crop ,score_threshold =args .score_threshold )#line:285
    with timer .env ('Copy'):#line:287
        OO0O0OOOOOO0OOOO0 ,O0OOOO000OO0OOOOO ,O000O0OOO0000OO0O ,O00OO0OO00O0OO000 =[OOOOO0OO0OOOO0000 [:args .top_k ]for OOOOO0OO0OOOO0000 in OO0OOOO0O0OOO0000 ]#line:288
        if isinstance (O0OOOO000OO0OOOOO ,list ):#line:289
            O0OOOO000OO0O0OO0 =O0OOOO000OO0OOOOO [0 ].cpu ().numpy ()#line:290
            OO00OO0OO0O0000OO =O0OOOO000OO0OOOOO [1 ].cpu ().numpy ()#line:291
        else :#line:292
            O0OOOO000OO0OOOOO =O0OOOO000OO0OOOOO .cpu ().numpy ()#line:293
        OO0O0OOOOOO0OOOO0 =OO0O0OOOOOO0OOOO0 .cpu ().numpy ()#line:294
        O000O0OOO0000OO0O =O000O0OOO0000OO0O .cpu ().numpy ()#line:295
        O00OO0OO00O0OO000 =O00OO0OO00O0OO000 .cpu ().numpy ()#line:296
    with timer .env ('Sync'):#line:298
        torch .cuda .synchronize ()#line:300
def prep_coco_cats ():#line:303
    ""#line:304
    for O0OO0OOOO0OOOO0OO ,O0O0O000O000OO00O in get_label_map ().items ():#line:305
        O0O00O0O0OOOO0000 =O0O0O000O000OO00O -1 #line:306
        coco_cats [O0O00O0O0OOOO0000 ]=O0OO0OOOO0OOOO0OO #line:307
        coco_cats_inv [O0OO0OOOO0OOOO0OO ]=O0O00O0O0OOOO0000 #line:308
def get_coco_cat (OO000O00OOO00000O ):#line:311
    ""#line:312
    return coco_cats [OO000O00OOO00000O ]#line:313
def get_transformed_cat (O0OOO0O0OOO0O000O ):#line:316
    ""#line:317
    return coco_cats_inv [O0OOO0O0OOO0O000O ]#line:318
class Detections :#line:321
    def __init__ (OO00OOOO0OOO0O0O0 ):#line:323
        OO00OOOO0OOO0O0O0 .bbox_data =[]#line:324
        OO00OOOO0OOO0O0O0 .mask_data =[]#line:325
    def add_bbox (OOO00OO0OO0OOOO0O ,O0OO00OOOO0O00OOO :int ,OO0OOO0O00OOOOO00 :int ,OOOOOOOO0OO0O0OO0 :list ,OOO0O0O0O00O0OO0O :float ):#line:327
        ""#line:328
        OOOOOOOO0OO0O0OO0 =[OOOOOOOO0OO0O0OO0 [0 ],OOOOOOOO0OO0O0OO0 [1 ],OOOOOOOO0OO0O0OO0 [2 ]-OOOOOOOO0OO0O0OO0 [0 ],OOOOOOOO0OO0O0OO0 [3 ]-OOOOOOOO0OO0O0OO0 [1 ]]#line:329
        OOOOOOOO0OO0O0OO0 =[round (float (O00O0O0000000O00O )*10 )/10 for O00O0O0000000O00O in OOOOOOOO0OO0O0OO0 ]#line:332
        OOO00OO0OO0OOOO0O .bbox_data .append ({'image_id':int (O0OO00OOOO0O00OOO ),'category_id':get_coco_cat (int (OO0OOO0O00OOOOO00 )),'bbox':OOOOOOOO0OO0O0OO0 ,'score':float (OOO0O0O0O00O0OO0O )})#line:339
    def add_mask (O0000OO0OOO0O000O ,OO000O0OO00OOOOO0 :int ,OOOO00O000O0O0OO0 :int ,O00O00O0OOO00000O :np .ndarray ,O0O00OOO00OOO000O :float ):#line:341
        ""#line:342
    def dump (O00O00OOO0000OOO0 ):#line:354
        OO0OO0OO000000000 =[(O00O00OOO0000OOO0 .bbox_data ,args .bbox_det_file ),(O00O00OOO0000OOO0 .mask_data ,args .mask_det_file )]#line:358
        for O0OOOOOO000000O0O ,O0OO0OO00O00O0OO0 in OO0OO0OO000000000 :#line:360
            with open (O0OO0OO00O00O0OO0 ,'w')as O000000O0O00OO0OO :#line:361
                json .dump (O0OOOOOO000000O0O ,O000000O0O00OO0OO )#line:362
    def dump_web (OO0OOOO0OOOO0O0O0 ):#line:364
        ""#line:365
        OOOO00O00O00O0O00 =['preserve_aspect_ratio','use_prediction_module','use_yolo_regressors','use_prediction_matching','train_masks']#line:368
        O0OO0000O0O00O000 ={'info':{'Config':{O00O000OOO00OO0O0 :getattr (cfg ,O00O000OOO00OO0O0 )for O00O000OOO00OO0O0 in OOOO00O00O00O0O00 },}}#line:374
        O000O00O00OOOOO0O =list (set ([O0O000O00OOOO00O0 ['image_id']for O0O000O00OOOO00O0 in OO0OOOO0OOOO0O0O0 .bbox_data ]))#line:376
        O000O00O00OOOOO0O .sort ()#line:377
        O00OOOOOOO00OOOOO ={_OO0OOO0000O00OO0O :O0O00000000O0O00O for O0O00000000O0O00O ,_OO0OOO0000O00OO0O in enumerate (O000O00O00OOOOO0O )}#line:378
        O0OO0000O0O00O000 ['images']=[{'image_id':OO00000000O0O0O00 ,'dets':[]}for OO00000000O0O0O00 in O000O00O00OOOOO0O ]#line:380
        for O0OO0OOO0OOOO0OOO ,OO0OOO0O00O0O0OO0 in zip (OO0OOOO0OOOO0O0O0 .bbox_data ,OO0OOOO0OOOO0O0O0 .mask_data ):#line:383
            O0OOO00OOO0000O0O =O0OO0000O0O00O000 ['images'][O00OOOOOOO00OOOOO [O0OO0OOO0OOOO0OOO ['image_id']]]#line:384
            O0OOO00OOO0000O0O ['dets'].append ({'score':O0OO0OOO0OOOO0OOO ['score'],'bbox':O0OO0OOO0OOOO0OOO ['bbox'],'category':cfg .dataset .class_names [get_transformed_cat (O0OO0OOO0OOOO0OOO ['category_id'])],'mask':OO0OOO0O00O0O0OO0 ['segmentation'],})#line:390
        with open (os .path .join (args .web_det_path ,'%s.json'%cfg .name ),'w')as OOOO0O0O0OO0OOO00 :#line:392
            json .dump (O0OO0000O0O00O000 ,OOOO0O0O0OO0OOO00 )#line:393
def _O0000OOOOOOOO00OO (OO00OO00OO000OOO0 ,OO00O000O0O0OOO0O ,iscrowd =False ):#line:396
    with timer .env ('Mask IoU'):#line:397
        OOOO0O0OOOO0000OO =mask_iou (OO00OO00OO000OOO0 ,OO00O000O0O0OOO0O ,iscrowd )#line:398
    return OOOO0O0OOOO0000OO .cpu ()#line:399
def _O00OOO00000OO0000 (OO0O00O0OO0OOO000 ,O0OOO0OOOOO000OOO ,iscrowd =False ):#line:402
    with timer .env ('BBox IoU'):#line:403
        OO000OOO0O0OO000O =jaccard (OO0O00O0OO0OOO000 ,O0OOO0OOOOO000OOO ,iscrowd )#line:404
    return OO000OOO0O0OO000O .cpu ()#line:405
def prep_metrics (OO0O0O000OO00OO0O ,O0OO00O0OOOOOO00O ,O0O000O00OO0O0OOO ,OO000O0OOOO0OOOO0 ,OOOOOOOOOOO000000 ,O0OOO0OOO0OO00OO0 ,O0O0O00000OO0O000 ,OO000OO0O000OOO0O ,OOO0O0O0OO0OO0O00 ,detections :Detections =None ):#line:408
    ""#line:409
    if not args .output_coco_json :#line:410
        with timer .env ('Prepare gt'):#line:411
            O000O0000OO000000 =torch .Tensor (OO000O0OOOO0OOOO0 [:,:4 ])#line:412
            O000O0000OO000000 [:,[0 ,2 ]]*=O0O0O00000OO0O000 #line:413
            O000O0000OO000000 [:,[1 ,3 ]]*=O0OOO0OOO0OO00OO0 #line:414
            OO0O00O00OO0O0OO0 =list (OO000O0OOOO0OOOO0 [:,4 ].astype (int ))#line:415
            OOOOOOOOOOO000000 =torch .Tensor (OOOOOOOOOOO000000 ).view (-1 ,O0OOO0OOO0OO00OO0 *O0O0O00000OO0O000 )#line:416
            if OO000OO0O000OOO0O >0 :#line:418
                O0O0OO0O0OOOO0OO0 =lambda O00OOOOO000OO00OO :(O00OOOOO000OO00OO [-OO000OO0O000OOO0O :],O00OOOOO000OO00OO [:-OO000OO0O000OOO0O ])#line:419
                OO00O000000OO0OO0 ,O000O0000OO000000 =O0O0OO0O0OOOO0OO0 (O000O0000OO000000 )#line:420
                O00O0OO0OO0O000OO ,OOOOOOOOOOO000000 =O0O0OO0O0OOOO0OO0 (OOOOOOOOOOO000000 )#line:421
                O00O0O000O00O0O0O ,OO0O00O00OO0O0OO0 =O0O0OO0O0OOOO0OO0 (OO0O00O00OO0O0OO0 )#line:422
    with timer .env ('Postprocess'):#line:424
        OOOOO0O00O0OO0OO0 ,O0O000OO00O0OOO00 ,O0O00O00O000O00OO ,OOOO0OO000OOO0OOO =postprocess (O0OO00O0OOOOOO00O ,O0O0O00000OO0O000 ,O0OOO0OOO0OO00OO0 ,crop_masks =args .crop ,score_threshold =args .score_threshold )#line:426
        if OOOOO0O00O0OO0OO0 .size (0 )==0 :#line:428
            return #line:429
        OOOOO0O00O0OO0OO0 =list (OOOOO0O00O0OO0OO0 .cpu ().numpy ().astype (int ))#line:431
        if isinstance (O0O000OO00O0OOO00 ,list ):#line:432
            O0OO0OO0O0OOOOO00 =list (O0O000OO00O0OOO00 [0 ].cpu ().numpy ().astype (float ))#line:433
            O00O00OO00000O00O =list (O0O000OO00O0OOO00 [1 ].cpu ().numpy ().astype (float ))#line:434
        else :#line:435
            O0O000OO00O0OOO00 =list (O0O000OO00O0OOO00 .cpu ().numpy ().astype (float ))#line:436
            O0OO0OO0O0OOOOO00 =O0O000OO00O0OOO00 #line:437
            O00O00OO00000O00O =O0O000OO00O0OOO00 #line:438
        OOOO0OO000OOO0OOO =OOOO0OO000OOO0OOO .view (-1 ,O0OOO0OOO0OO00OO0 *O0O0O00000OO0O000 ).cuda ()#line:439
        O0O00O00O000O00OO =O0O00O00O000O00OO .cuda ()#line:440
    if args .output_coco_json :#line:442
        with timer .env ('JSON Output'):#line:443
            O0O00O00O000O00OO =O0O00O00O000O00OO .cpu ().numpy ()#line:444
            OOOO0OO000OOO0OOO =OOOO0OO000OOO0OOO .view (-1 ,O0OOO0OOO0OO00OO0 ,O0O0O00000OO0O000 ).cpu ().numpy ()#line:445
            for O00O00O0000000O0O in range (OOOO0OO000OOO0OOO .shape [0 ]):#line:446
                if (O0O00O00O000O00OO [O00O00O0000000O0O ,3 ]-O0O00O00O000O00OO [O00O00O0000000O0O ,1 ])*(O0O00O00O000O00OO [O00O00O0000000O0O ,2 ]-O0O00O00O000O00OO [O00O00O0000000O0O ,0 ])>0 :#line:448
                    detections .add_bbox (OOO0O0O0OO0OO0O00 ,OOOOO0O00O0OO0OO0 [O00O00O0000000O0O ],O0O00O00O000O00OO [O00O00O0000000O0O ,:],O0OO0OO0O0OOOOO00 [O00O00O0000000O0O ])#line:449
                    detections .add_mask (OOO0O0O0OO0OO0O00 ,OOOOO0O00O0OO0OO0 [O00O00O0000000O0O ],OOOO0OO000OOO0OOO [O00O00O0000000O0O ,:,:],O00O00OO00000O00O [O00O00O0000000O0O ])#line:450
            return #line:451
    with timer .env ('Eval Setup'):#line:453
        OO0OOOOO0OOO0000O =len (OOOOO0O00O0OO0OO0 )#line:454
        O0OO0O0O0000O0O0O =len (OO0O00O00OO0O0OO0 )#line:455
        O0OO00000O0000O00 =_O0000OOOOOOOO00OO (OOOO0OO000OOO0OOO ,OOOOOOOOOOO000000 )#line:457
        O000OO00OOOO0OO0O =_O00OOO00000OO0000 (O0O00O00O000O00OO .float (),O000O0000OO000000 .float ())#line:458
        if OO000OO0O000OOO0O >0 :#line:460
            O0OO0000O0OO000O0 =_O0000OOOOOOOO00OO (OOOO0OO000OOO0OOO ,O00O0OO0OO0O000OO ,iscrowd =True )#line:461
            O0OOO00000O0O00OO =_O00OOO00000OO0000 (O0O00O00O000O00OO .float (),OO00O000000OO0OO0 .float (),iscrowd =True )#line:462
        else :#line:463
            O0OO0000O0OO000O0 =None #line:464
            O0OOO00000O0O00OO =None #line:465
        O00OO0OOOO0O0OOOO =sorted (range (OO0OOOOO0OOO0000O ),key =lambda OO0OOOOO0O0OOO0OO :-O0OO0OO0O0OOOOO00 [OO0OOOOO0O0OOO0OO ])#line:467
        OO00OOOOOOOO0000O =sorted (O00OO0OOOO0O0OOOO ,key =lambda O00000000OO0OO0O0 :-O00O00OO00000O00O [O00000000OO0OO0O0 ])#line:468
        OO00OOO0OO0O00OO0 =[('box',lambda O000OO0OOO0O0OOO0 ,OO00000OO0OOO0OO0 :O000OO00OOOO0OO0O [O000OO0OOO0O0OOO0 ,OO00000OO0OOO0OO0 ].item (),lambda OOO00OOOOO00OOOOO ,OOOO00OOOO0000OO0 :O0OOO00000O0O00OO [OOO00OOOOO00OOOOO ,OOOO00OOOO0000OO0 ].item (),lambda O0OOO0000O0OO00O0 :O0OO0OO0O0OOOOO00 [O0OOO0000O0OO00O0 ],O00OO0OOOO0O0OOOO ),('mask',lambda OO00OOOOOOO0000O0 ,O0OOOO0O00O0O00OO :O0OO00000O0000O00 [OO00OOOOOOO0000O0 ,O0OOOO0O00O0O00OO ].item (),lambda OO0OO0O0O00O0O00O ,O000O00O00OOO0OOO :O0OO0000O0OO000O0 [OO0OO0O0O00O0O00O ,O000O00O00OOO0OOO ].item (),lambda O0OOO0OO0O00O00OO :O00O00OO00000O00O [O0OOO0OO0O00O00OO ],OO00OOOOOOOO0000O )]#line:477
    timer .start ('Main loop')#line:479
    for _OO000O0O0O0O0O0O0 in set (OOOOO0O00O0OO0OO0 +OO0O00O00OO0O0OO0 ):#line:480
        O00OOOOOO0O000OOO =[]#line:481
        O00O000O000000O00 =sum ([1 for O0O0OO0OO0OO00000 in OO0O00O00OO0O0OO0 if O0O0OO0OO0OO00000 ==_OO000O0O0O0O0O0O0 ])#line:482
        for OO0O0OOOOOO000OOO in range (len (iou_thresholds )):#line:484
            OOOO0OOO0O0OO000O =iou_thresholds [OO0O0OOOOOO000OOO ]#line:485
            for OOO000OOO0OO00000 ,OO0O000OO00OO00OO ,OO000O00OO0OOO0O0 ,OO00O0O0O0OO0O00O ,OO0000OOO0OOOOOOO in OO00OOO0OO0O00OO0 :#line:487
                O0O000O0O00OOO00O =[False ]*len (OO0O00O00OO0O0OO0 )#line:488
                OO0OO00O00OO0000O =OO0O0O000OO00OO0O [OOO000OOO0OO00000 ][OO0O0OOOOOO000OOO ][_OO000O0O0O0O0O0O0 ]#line:490
                OO0OO00O00OO0000O .add_gt_positives (O00O000O000000O00 )#line:491
                for O00O00O0000000O0O in OO0000OOO0OOOOOOO :#line:493
                    if OOOOO0O00O0OO0OO0 [O00O00O0000000O0O ]!=_OO000O0O0O0O0O0O0 :#line:494
                        continue #line:495
                    OO00O0OO0000O000O =OOOO0OOO0O0OO000O #line:497
                    OO0OO0O000OOOOOOO =-1 #line:498
                    for O0OO0OO0O00O0O0OO in range (O0OO0O0O0000O0O0O ):#line:499
                        if O0O000O0O00OOO00O [O0OO0OO0O00O0O0OO ]or OO0O00O00OO0O0OO0 [O0OO0OO0O00O0O0OO ]!=_OO000O0O0O0O0O0O0 :#line:500
                            continue #line:501
                        O000O0OOO0O0OO0O0 =OO0O000OO00OO00OO (O00O00O0000000O0O ,O0OO0OO0O00O0O0OO )#line:503
                        if O000O0OOO0O0OO0O0 >OO00O0OO0000O000O :#line:505
                            OO00O0OO0000O000O =O000O0OOO0O0OO0O0 #line:506
                            OO0OO0O000OOOOOOO =O0OO0OO0O00O0O0OO #line:507
                    if OO0OO0O000OOOOOOO >=0 :#line:509
                        O0O000O0O00OOO00O [OO0OO0O000OOOOOOO ]=True #line:510
                        OO0OO00O00OO0000O .push (OO00O0O0O0OO0O00O (O00O00O0000000O0O ),True )#line:511
                    else :#line:512
                        O00O0OOO0OO0O0OOO =False #line:514
                        if OO000OO0O000OOO0O >0 :#line:516
                            for O0OO0OO0O00O0O0OO in range (len (O00O0O000O00O0O0O )):#line:517
                                if O00O0O000O00O0O0O [O0OO0OO0O00O0O0OO ]!=_OO000O0O0O0O0O0O0 :#line:518
                                    continue #line:519
                                O000O0OOO0O0OO0O0 =OO000O00OO0OOO0O0 (O00O00O0000000O0O ,O0OO0OO0O00O0O0OO )#line:521
                                if O000O0OOO0O0OO0O0 >OOOO0OOO0O0OO000O :#line:523
                                    O00O0OOO0OO0O0OOO =True #line:524
                                    break #line:525
                        if not O00O0OOO0OO0O0OOO :#line:530
                            OO0OO00O00OO0000O .push (OO00O0O0O0OO0O00O (O00O00O0000000O0O ),False )#line:531
    timer .stop ('Main loop')#line:532
class APDataObject :#line:535
    ""#line:539
    def __init__ (O000OO0O0OO00OO0O ):#line:541
        O000OO0O0OO00OO0O .data_points =[]#line:542
        O000OO0O0OO00OO0O .num_gt_positives =0 #line:543
    def push (OO0000OOO00OOOOO0 ,O00OOOO00OO0O0O00 :float ,OO0OO0O0O0O00OO00 :bool ):#line:545
        OO0000OOO00OOOOO0 .data_points .append ((O00OOOO00OO0O0O00 ,OO0OO0O0O0O00OO00 ))#line:546
    def add_gt_positives (O0O00O0O000OOOOO0 ,O0OO000O0OO0O00OO :int ):#line:548
        ""#line:549
        O0O00O0O000OOOOO0 .num_gt_positives +=O0OO000O0OO0O00OO #line:550
    def is_empty (OO000OOO00O000O00 )->bool :#line:552
        return len (OO000OOO00O000O00 .data_points )==0 and OO000OOO00O000O00 .num_gt_positives ==0 #line:553
    def get_ap (OOOO00OOO0O0O0OO0 )->float :#line:555
        ""#line:556
        if OOOO00OOO0O0O0OO0 .num_gt_positives ==0 :#line:558
            return 0 #line:559
        OOOO00OOO0O0O0OO0 .data_points .sort (key =lambda OO0OO00O0OOOO00O0 :-OO0OO00O0OOOO00O0 [0 ])#line:562
        OOOO00000OOO0OOOO =[]#line:564
        OO0OOOOOOOOOO0O0O =[]#line:565
        OOO0O00O00O0O00O0 =0 #line:566
        O000OOO00O0O00000 =0 #line:567
        for OOOOO0000O0000O00 in OOOO00OOO0O0O0OO0 .data_points :#line:570
            if OOOOO0000O0000O00 [1 ]:#line:572
                OOO0O00O00O0O00O0 +=1 #line:573
            else :#line:574
                O000OOO00O0O00000 +=1 #line:575
            O000OO0OO000O0O0O =OOO0O00O00O0O00O0 /(OOO0O00O00O0O00O0 +O000OOO00O0O00000 )#line:577
            OO000OOOO00O0O0O0 =OOO0O00O00O0O00O0 /OOOO00OOO0O0O0OO0 .num_gt_positives #line:578
            OOOO00000OOO0OOOO .append (O000OO0OO000O0O0O )#line:580
            OO0OOOOOOOOOO0O0O .append (OO000OOOO00O0O0O0 )#line:581
        for OO0O000O0OO0OOO0O in range (len (OOOO00000OOO0OOOO )-1 ,0 ,-1 ):#line:586
            if OOOO00000OOO0OOOO [OO0O000O0OO0OOO0O ]>OOOO00000OOO0OOOO [OO0O000O0OO0OOO0O -1 ]:#line:587
                OOOO00000OOO0OOOO [OO0O000O0OO0OOO0O -1 ]=OOOO00000OOO0OOOO [OO0O000O0OO0OOO0O ]#line:588
        OOOOOO0O0O0O00OOO =[0 ]*101 #line:591
        O000OO0OO0O0O0OOO =np .array ([O00O0O00000OOO0OO /100 for O00O0O00000OOO0OO in range (101 )])#line:592
        OO0OOOOOOOOOO0O0O =np .array (OO0OOOOOOOOOO0O0O )#line:593
        OOOO00O0O0O0OO00O =np .searchsorted (OO0OOOOOOOOOO0O0O ,O000OO0OO0O0O0OOO ,side ='left')#line:598
        for OOOO0000OO000O0O0 ,OOOO0O0O0OO0000O0 in enumerate (OOOO00O0O0O0OO00O ):#line:599
            if OOOO0O0O0OO0000O0 <len (OOOO00000OOO0OOOO ):#line:600
                OOOOOO0O0O0O00OOO [OOOO0000OO000O0O0 ]=OOOO00000OOO0OOOO [OOOO0O0O0OO0000O0 ]#line:601
        return sum (OOOOOO0O0O0O00OOO )/len (OOOOOO0O0O0O00OOO )#line:605
def badhash (O0O00OO0OOOO000O0 ):#line:608
    ""#line:614
    O0O00OO0OOOO000O0 =(((O0O00OO0OOOO000O0 >>16 )^O0O00OO0OOOO000O0 )*0x045d9f3b )&0xFFFFFFFF #line:615
    O0O00OO0OOOO000O0 =(((O0O00OO0OOOO000O0 >>16 )^O0O00OO0OOOO000O0 )*0x045d9f3b )&0xFFFFFFFF #line:616
    O0O00OO0OOOO000O0 =((O0O00OO0OOOO000O0 >>16 )^O0O00OO0OOOO000O0 )&0xFFFFFFFF #line:617
    return O0O00OO0OOOO000O0 #line:618
def evalimage_KDG (O0OOO0O0O0O0O00OO :Yolact ,O0O00O0000O00000O :str ,save_path :str =None ):#line:620
    O000OOO0OO000000O =cv2 .imread (O0O00O0000O00000O )#line:621
    O00O00O0OO00O0OO0 =torch .from_numpy (cv2 .imread (O0O00O0000O00000O )).cuda ().float ()#line:622
    OOO00000OO0O0O0O0 =FastBaseTransform ()(O00O00O0OO00O0OO0 .unsqueeze (0 ))#line:623
    O00OO0OOOO0OOO0O0 =O0OOO0O0O0O0O00OO (OOO00000OO0O0O0O0 )#line:624
    OO0O000O000O0O0O0 =prep_display (O00OO0OOOO0OOO0O0 ,O00O00O0OO00O0OO0 ,None ,None ,undo_transform =False )#line:626
    if save_path is None :#line:628
        OO0O000O000O0O0O0 =OO0O000O000O0O0O0 [:,:,(2 ,1 ,0 )]#line:629
    cv2 .imwrite (save_path ,OO0O000O000O0O0O0 )#line:631
    os .remove (O0O00O0000O00000O )#line:632
def evalimage (O0OO00O00000OOOO0 :Yolact ,OOO00000O0O0OOO0O :str ,save_path :str =None ):#line:635
    OOOOO0OO0O000000O =cv2 .imread (OOO00000O0O0OOO0O )#line:636
    OO0O0O00000O0O000 =torch .from_numpy (cv2 .imread (OOO00000O0O0OOO0O )).cuda ().float ()#line:637
    OO000O00OOO000OO0 =FastBaseTransform ()(OO0O0O00000O0O000 .unsqueeze (0 ))#line:638
    OOOOO00OO000000O0 =O0OO00O00000OOOO0 (OO000O00OOO000OO0 )#line:639
    OOOO0OOO000O000O0 =prep_display (OOOOO00OO000000O0 ,OO0O0O00000O0O000 ,None ,None ,undo_transform =False )#line:641
    if save_path is None :#line:644
        OOOO0OOO000O000O0 =OOOO0OOO000O000O0 [:,:,(2 ,1 ,0 )]#line:645
    if save_path is None :#line:647
        plt .imshow (OOOO0OOO000O000O0 )#line:648
        plt .title (OOO00000O0O0OOO0O )#line:649
        plt .show ()#line:650
    else :#line:651
        cv2 .imwrite (save_path ,OOOO0OOO000O000O0 )#line:652
def evalimages_KDG (OO0000O0000000000 :Yolact ,O0O0000OO0O00OO0O :str ,OOO0O0O0O00O00OOO :str ):#line:657
    if not os .path .exists (OOO0O0O0O00O00OOO ):#line:658
        os .mkdir (OOO0O0O0O00O00OOO )#line:659
    while True :#line:661
        print ()#line:662
        for OOOO0OOO00OO00O00 in Path (O0O0000OO0O00OO0O ).glob ('*'):#line:663
            O0O0000000000OOOO =time .time ()#line:664
            OO000O0O00O00O0O0 =str (OOOO0OOO00OO00O00 )#line:666
            O0OO0OOOOOO0OO0OO =os .path .basename (OO000O0O00O00O0O0 )#line:667
            O0OO0OOOOOO0OO0OO ='.'.join (O0OO0OOOOOO0OO0OO .split ('.')[:-1 ])+'.jpg'#line:668
            OOOO0OO0OOO00O0OO =os .path .join (OOO0O0O0O00O00OOO ,O0OO0OOOOOO0OO0OO )#line:669
            evalimage_KDG (OO0000O0000000000 ,OO000O0O00O00O0O0 ,OOOO0OO0OOO00O0OO )#line:671
            print (OO000O0O00O00O0O0 +' -> '+OOOO0OO0OOO00O0OO )#line:672
            print ('coordinate [ '+str (g_x1 )+' , '+str (g_x2 )+' , '+str (g_y1 )+' , '+str (g_y2 )+' ]')#line:673
            print ("--- %s seconds ---"%(time .time ()-O0O0000000000OOOO ))#line:676
        time .sleep (3 )#line:678
        O0O0OO0OOO0OOO000 =datetime .datetime .now ()#line:679
        print (O0O0OO0OOO0OOO000 )#line:680
def evalimages (OOOOOOO000000OOOO :Yolact ,O0OO00000O00OO0O0 :str ,O0OO00OO000OOO0OO :str ):#line:683
    if not os .path .exists (O0OO00OO000OOO0OO ):#line:684
        os .mkdir (O0OO00OO000OOO0OO )#line:685
    print ()#line:687
    for O00OOOOOO0O000000 in Path (O0OO00000O00OO0O0 ).glob ('*'):#line:688
        OO00OOO000O00OOOO =str (O00OOOOOO0O000000 )#line:689
        OOO0OO0000O0O0O00 =os .path .basename (OO00OOO000O00OOOO )#line:690
        OOO0OO0000O0O0O00 ='.'.join (OOO0OO0000O0O0O00 .split ('.')[:-1 ])+'.png'#line:691
        OO0O0O0OO000OO0O0 =os .path .join (O0OO00OO000OOO0OO ,OOO0OO0000O0O0O00 )#line:692
        evalimage (OOOOOOO000000OOOO ,OO00OOO000O00OOOO ,OO0O0O0OO000OO0O0 )#line:694
        print (OO00OOO000O00OOOO +' -> '+OO0O0O0OO000OO0O0 )#line:695
    print ('Done.')#line:696
from multiprocessing .pool import ThreadPool #line:699
from queue import Queue #line:700
class CustomDataParallel (torch .nn .DataParallel ):#line:703
    ""#line:704
    def gather (OOOOO00OOO000000O ,OO0OO00OO000O000O ,O0O0000O0OOO0OOOO ):#line:706
        return sum (OO0OO00OO000O000O ,[])#line:708
def evalvideo (O00OO0OOO0O0OO0OO :Yolact ,O00O0O0OOO000O0OO :str ,out_path :str =None ):#line:711
    O0OO0OO000OO000OO =O00O0O0OOO000O0OO .isdigit ()#line:713
    cudnn .benchmark =True #line:716
    if O0OO0OO000OO000OO :#line:718
        O00OOOOO00O0O0OOO =cv2 .VideoCapture (int (O00O0O0OOO000O0OO ))#line:719
    else :#line:720
        O00OOOOO00O0O0OOO =cv2 .VideoCapture (O00O0O0OOO000O0OO )#line:721
    if not O00OOOOO00O0O0OOO .isOpened ():#line:723
        print ('Could not open video "%s"'%O00O0O0OOO000O0OO )#line:724
        exit (-1 )#line:725
    OO000OOOOO00O0OO0 =round (O00OOOOO00O0O0OOO .get (cv2 .CAP_PROP_FPS ))#line:727
    OOOOO0OOOO00000OO =round (O00OOOOO00O0O0OOO .get (cv2 .CAP_PROP_FRAME_WIDTH ))#line:728
    OOOOO00O00OO00O00 =round (O00OOOOO00O0O0OOO .get (cv2 .CAP_PROP_FRAME_HEIGHT ))#line:729
    if O0OO0OO000OO000OO :#line:731
        OO000O0O000OOO0OO =float ('inf')#line:732
    else :#line:733
        OO000O0O000OOO0OO =round (O00OOOOO00O0O0OOO .get (cv2 .CAP_PROP_FRAME_COUNT ))#line:734
    O00OO0OOO0O0OO0OO =CustomDataParallel (O00OO0OOO0O0OO0OO ).cuda ()#line:736
    O000OOO000OO0O0O0 =torch .nn .DataParallel (FastBaseTransform ()).cuda ()#line:737
    O00OO0O0O0OO0O0OO =MovingAverage (100 )#line:738
    O0000O0OO0OOO000O =0 #line:739
    OO000O000OOOO0O00 =1 /OO000OOOOO00O0OO0 #line:740
    OOO0OOOOO00OO00O0 =True #line:741
    O000O0OO0O0OO0000 =''#line:742
    O00O0O00OOO000OO0 =False #line:743
    OO00OOO000O0OOOOO =0 #line:744
    if out_path is not None :#line:746
        OOO000OOOOO00O0OO =cv2 .VideoWriter (out_path ,cv2 .VideoWriter_fourcc (*"mp4v"),OO000OOOOO00O0OO0 ,(OOOOO0OOOO00000OO ,OOOOO00O00OO00O00 ))#line:747
    def O0O000OOOOO00000O ():#line:749
        print ()#line:750
        OO0O00O0OOOOO0OO0 .terminate ()#line:751
        O00OOOOO00O0O0OOO .release ()#line:752
        if out_path is not None :#line:753
            OOO000OOOOO00O0OO .release ()#line:754
        cv2 .destroyAllWindows ()#line:755
        exit ()#line:756
    def O0OOO0000000O000O (OOOOO0O0O0OOOOO00 ):#line:758
        OOO0O0000O00OO0O0 =[]#line:759
        for O00OO000O00O00O0O in range (args .video_multiframe ):#line:760
            OO0OOOOOO00O00O0O =OOOOO0O0O0OOOOO00 .read ()[1 ]#line:761
            if OO0OOOOOO00O00O0O is None :#line:762
                return OOO0O0000O00OO0O0 #line:763
            OOO0O0000O00OO0O0 .append (OO0OOOOOO00O00O0O )#line:764
        return OOO0O0000O00OO0O0 #line:765
    def OOO00OOO0O0OOOOO0 (OO0OO0OOOOO0000OO ):#line:767
        with torch .no_grad ():#line:768
            OO0OO0OOOOO0000OO =[torch .from_numpy (OOOOOO00O00O0000O ).cuda ().float ()for OOOOOO00O00O0000O in OO0OO0OOOOO0000OO ]#line:769
            return OO0OO0OOOOO0000OO ,O000OOO000OO0O0O0 (torch .stack (OO0OO0OOOOO0000OO ,0 ))#line:770
    def OOOOO00000000O0OO (O00O0O00O0OO00000 ):#line:772
        with torch .no_grad ():#line:773
            OO00OOOOOO000O0OO ,OOOO0OOO000OO0000 =O00O0O00O0OO00000 #line:774
            OOO0OOO0O00OOOO0O =0 #line:775
            while OOOO0OOO000OO0000 .size (0 )<args .video_multiframe :#line:776
                OOOO0OOO000OO0000 =torch .cat ([OOOO0OOO000OO0000 ,OOOO0OOO000OO0000 [0 ].unsqueeze (0 )],dim =0 )#line:777
                OOO0OOO0O00OOOO0O +=1 #line:778
            O000OO00O0O00OOO0 =O00OO0OOO0O0OO0OO (OOOO0OOO000OO0000 )#line:779
            if OOO0OOO0O00OOOO0O >0 :#line:780
                O000OO00O0O00OOO0 =O000OO00O0O00OOO0 [:-OOO0OOO0O00OOOO0O ]#line:781
            return OO00OOOOOO000O0OO ,O000OO00O0O00OOO0 #line:782
    def O0O0OO0OO0OOOO00O (OOO00OO0OOO0000O0 ,O0000000000O00O00 ):#line:784
        with torch .no_grad ():#line:785
            O00O0OO0O000OO0OO ,O0OO0OO000OO0OO00 =OOO00OO0OOO0000O0 #line:786
            return prep_display (O0OO0OO000OO0OO00 ,O00O0OO0O000OO0OO ,None ,None ,undo_transform =False ,class_color =True ,fps_str =O0000000000O00O00 )#line:787
    O000O0OO0O00OO0O0 =Queue ()#line:789
    OO0OOOO0OO00OOOO0 =0 #line:790
    def O0O00O0O00O0O0000 ():#line:793
        try :#line:794
            nonlocal O000O0OO0O00OO0O0 ,OOO0OOOOO00OO00O0 ,OO0OOOO0OO00OOOO0 ,O0OO0OO000OO000OO ,OO000O0O000OOO0OO ,OO00OOO000O0OOOOO ,O00O0O00OOO000OO0 #line:795
            O0OOOOO00OOO0OO0O =MovingAverage (100 )#line:797
            O0OOO0OO0O0OOOO00 =OO000O000OOOO0O00 #line:798
            OO00O0O00OOOOO0O0 =None #line:799
            O0O00OO0OO0O0OO0O =0.0005 #line:800
            OOOO00OO0O00000OO =ProgressBar (30 ,OO000O0O000OOO0OO )#line:801
            while OOO0OOOOO00OO00O0 :#line:803
                OO00O00OO0OO00OOO =time .time ()#line:804
                if not O000O0OO0O00OO0O0 .empty ():#line:806
                    OO0O0OOO0O0000O00 =time .time ()#line:807
                    if OO00O0O00OOOOO0O0 is not None :#line:808
                        O0OOOOO00OOO0OO0O .add (OO0O0OOO0O0000O00 -OO00O0O00OOOOO0O0 )#line:809
                        OO0OOOO0OO00OOOO0 =1 /O0OOOOO00OOO0OO0O .get_avg ()#line:810
                    if out_path is None :#line:811
                        cv2 .imshow (O00O0O0OOO000O0OO ,O000O0OO0O00OO0O0 .get ())#line:812
                    else :#line:813
                        OOO000OOOOO00O0OO .write (O000O0OO0O00OO0O0 .get ())#line:814
                    OO00OOO000O0OOOOO +=1 #line:815
                    OO00O0O00OOOOO0O0 =OO0O0OOO0O0000O00 #line:816
                    if out_path is not None :#line:818
                        if O0OOOOO00OOO0OO0O .get_avg ()==0 :#line:819
                            OO0OO0OOO000O00OO =0 #line:820
                        else :#line:821
                            OO0OO0OOO000O00OO =1 /O0OOOOO00OOO0OO0O .get_avg ()#line:822
                        O00O0OO000OOO00OO =OO00OOO000O0OOOOO /OO000O0O000OOO0OO *100 #line:823
                        OOOO00OO0O00000OO .set_val (OO00OOO000O0OOOOO )#line:824
                        print ('\rProcessing Frames  %s %6d / %6d (%5.2f%%)    %5.2f fps        '%(repr (OOOO00OO0O00000OO ),OO00OOO000O0OOOOO ,OO000O0O000OOO0OO ,O00O0OO000OOO00OO ,OO0OO0OOO000O00OO ),end ='')#line:827
                if out_path is None and cv2 .waitKey (1 )==27 :#line:830
                    OOO0OOOOO00OO00O0 =False #line:832
                if not (OO00OOO000O0OOOOO <OO000O0O000OOO0OO ):#line:833
                    OOO0OOOOO00OO00O0 =False #line:834
                if not O00O0O00OOO000OO0 :#line:836
                    OOO00OO000OOO0OOO =O000O0OO0O00OO0O0 .qsize ()#line:837
                    if OOO00OO000OOO0OOO <args .video_multiframe :#line:838
                        O0OOO0OO0O0OOOO00 +=O0O00OO0OO0O0OO0O #line:839
                    elif OOO00OO000OOO0OOO >args .video_multiframe :#line:840
                        O0OOO0OO0O0OOOO00 -=O0O00OO0OO0O0OO0O #line:841
                        if O0OOO0OO0O0OOOO00 <0 :#line:842
                            O0OOO0OO0O0OOOO00 =0 #line:843
                    O0O000OOOO0OOOO0O =O0OOO0OO0O0OOOO00 if O0OO0OO000OO000OO else max (O0OOO0OO0O0OOOO00 ,OO000O000OOOO0O00 )#line:845
                else :#line:846
                    O0O000OOOO0OOOO0O =OO000O000OOOO0O00 #line:847
                OO0O0O0OO0OO0OOO0 =max (2 *O0O000OOOO0OOOO0O -O0OOOOO00OOO0OO0O .get_avg (),0 )#line:849
                O0OOOO00O000O00OO =OO00O00OO0OO00OOO +OO0O0O0OO0OO0OOO0 -0.001 #line:850
                if out_path is None or args .emulate_playback :#line:852
                    while time .time ()<O0OOOO00O000O00OO :#line:854
                        time .sleep (0.001 )#line:855
                else :#line:856
                    time .sleep (0.001 )#line:858
        except :#line:859
            import traceback #line:861
            traceback .print_exc ()#line:862
    O0O0OOOO0O0O00000 =lambda OO0O0O0OOOOOO0O00 ,OOO0OO0O00OOOOO00 :(OO0O0O0OOOOOO0O00 [0 ][OOO0OO0O00OOOOO00 ]if OO0O0O0OOOOOO0O00 [1 ][OOO0OO0O00OOOOO00 ]['detection']is None else OO0O0O0OOOOOO0O00 [0 ][OOO0OO0O00OOOOO00 ].to (OO0O0O0OOOOOO0O00 [1 ][OOO0OO0O00OOOOO00 ]['detection']['box'].device ),[OO0O0O0OOOOOO0O00 [1 ][OOO0OO0O00OOOOO00 ]])#line:865
    print ('Initializing model... ',end ='')#line:868
    OOOOOOOO00O00O00O =OOOOO00000000O0OO (OOO00OOO0O0OOOOO0 (O0OOO0000000O000O (O00OOOOO00O0O0OOO )))#line:869
    print ('Done.')#line:870
    OO0O000OOOOOO00OO =[O0O0OO0OO0OOOO00O ,OOOOO00000000O0OO ,OOO00OOO0O0OOOOO0 ]#line:873
    OO0O00O0OOOOO0OO0 =ThreadPool (processes =len (OO0O000OOOOOO00OO )+args .video_multiframe +2 )#line:874
    OO0O00O0OOOOO0OO0 .apply_async (O0O00O0O00O0O0000 )#line:875
    OO0O0OOO000O0OOO0 =[{'value':O0O0OOOO0O0O00000 (OOOOOOOO00O00O00O ,OOO00O0O00000O00O ),'idx':0 }for OOO00O0O00000O00O in range (len (OOOOOOOO00O00O00O [0 ]))]#line:876
    print ()#line:878
    if out_path is None :print ('Press Escape to close.')#line:879
    try :#line:880
        while O00OOOOO00O0O0OOO .isOpened ()and OOO0OOOOO00OO00O0 :#line:881
            while O000O0OO0O00OO0O0 .qsize ()>100 :#line:883
                time .sleep (0.001 )#line:884
            O0O0000O000000000 =time .time ()#line:886
            if not O00O0O00OOO000OO0 :#line:889
                OOOOO00OO00OO0OO0 =OO0O00O0OOOOO0OO0 .apply_async (O0OOO0000000O000O ,args =(O00OOOOO00O0O0OOO ,))#line:890
            else :#line:891
                OOOOO00OO00OO0OO0 =None #line:892
            if not (O00O0O00OOO000OO0 and len (OO0O0OOO000O0OOO0 )==0 ):#line:894
                for OOO000OOOOO0OO00O in OO0O0OOO000O0OOO0 :#line:897
                    _OOOOO0OOO0O0O00O0 =[OOO000OOOOO0OO00O ['value']]#line:898
                    if OOO000OOOOO0OO00O ['idx']==0 :#line:899
                        _OOOOO0OOO0O0O00O0 .append (O000O0OO0O0OO0000 )#line:900
                    OOO000OOOOO0OO00O ['value']=OO0O00O0OOOOO0OO0 .apply_async (OO0O000OOOOOO00OO [OOO000OOOOO0OO00O ['idx']],args =_OOOOO0OOO0O0O00O0 )#line:901
                for OOO000OOOOO0OO00O in OO0O0OOO000O0OOO0 :#line:904
                    if OOO000OOOOO0OO00O ['idx']==0 :#line:905
                        O000O0OO0O00OO0O0 .put (OOO000OOOOO0OO00O ['value'].get ())#line:906
                OO0O0OOO000O0OOO0 =[O0O0OOOOOOO0O0O0O for O0O0OOOOOOO0O0O0O in OO0O0OOO000O0OOO0 if O0O0OOOOOOO0O0O0O ['idx']>0 ]#line:909
                for OOO000OOOOO0OO00O in list (reversed (OO0O0OOO000O0OOO0 )):#line:912
                    OOO000OOOOO0OO00O ['value']=OOO000OOOOO0OO00O ['value'].get ()#line:913
                    OOO000OOOOO0OO00O ['idx']-=1 #line:914
                    if OOO000OOOOO0OO00O ['idx']==0 :#line:916
                        OO0O0OOO000O0OOO0 +=[{'value':O0O0OOOO0O0O00000 (OOO000OOOOO0OO00O ['value'],OO00OO0000000O00O ),'idx':0 }for OO00OO0000000O00O in range (1 ,len (OOO000OOOOO0OO00O ['value'][0 ]))]#line:919
                        OOO000OOOOO0OO00O ['value']=O0O0OOOO0O0O00000 (OOO000OOOOO0OO00O ['value'],0 )#line:920
                if OOOOO00OO00OO0OO0 is not None :#line:923
                    OO0O00O0O00OO0O0O =OOOOO00OO00OO0OO0 .get ()#line:924
                    if len (OO0O00O0O00OO0O0O )==0 :#line:925
                        O00O0O00OOO000OO0 =True #line:926
                    else :#line:927
                        OO0O0OOO000O0OOO0 .append ({'value':OO0O00O0O00OO0O0O ,'idx':len (OO0O000OOOOOO00OO )-1 })#line:928
                O00OO0O0O0OO0O0OO .add (time .time ()-O0O0000O000000000 )#line:931
                O0000O0OO0OOO000O =args .video_multiframe /O00OO0O0O0OO0O0OO .get_avg ()#line:932
            else :#line:933
                O0000O0OO0OOO000O =0 #line:934
            O000O0OO0O0OO0000 ='Processing FPS: %.2f | Video Playback FPS: %.2f | Frames in Buffer: %d'%(O0000O0OO0OOO000O ,OO0OOOO0OO00OOOO0 ,O000O0OO0O00OO0O0 .qsize ())#line:937
            if not args .display_fps :#line:938
                print ('\r'+O000O0OO0O0OO0000 +'    ',end ='')#line:939
    except KeyboardInterrupt :#line:941
        print ('\nStopping...')#line:942
    O0O000OOOOO00000O ()#line:944
def evaluate_KDG (O0000000OO00O00OO :Yolact ,O0O0OOO00000000O0 ,train_mode =False ):#line:948
    O0000000OO00O00OO .detect .use_fast_nms =args .fast_nms #line:949
    O0000000OO00O00OO .detect .use_cross_class_nms =args .cross_class_nms #line:950
    cfg .mask_proto_debug =args .mask_proto_debug #line:951
    if args .images is not None :#line:953
        O00OOOOOO0000OOOO ,OOO00O0O0OOOOOO0O =args .images .split (':')#line:954
        evalimages_KDG (O0000000OO00O00OO ,O00OOOOOO0000OOOO ,OOO00O0O0OOOOOO0O )#line:955
def evaluate (O000O0O000O00O0OO :Yolact ,O0000O000O00O0000 ,train_mode =False ):#line:959
    O000O0O000O00O0OO .detect .use_fast_nms =args .fast_nms #line:960
    O000O0O000O00O0OO .detect .use_cross_class_nms =args .cross_class_nms #line:961
    cfg .mask_proto_debug =args .mask_proto_debug #line:962
    if args .image is not None :#line:965
        if ':'in args .image :#line:966
            OO0000O0O00000000 ,OOO00OOO0O0O00OO0 =args .image .split (':')#line:967
            evalimage (O000O0O000O00O0OO ,OO0000O0O00000000 ,OOO00OOO0O0O00OO0 )#line:968
        else :#line:969
            evalimage (O000O0O000O00O0OO ,args .image )#line:970
        return #line:971
    elif args .images is not None :#line:972
        OO0000O0O00000000 ,OOO00OOO0O0O00OO0 =args .images .split (':')#line:973
        evalimages (O000O0O000O00O0OO ,OO0000O0O00000000 ,OOO00OOO0O0O00OO0 )#line:974
        return #line:975
    elif args .video is not None :#line:976
        if ':'in args .video :#line:977
            OO0000O0O00000000 ,OOO00OOO0O0O00OO0 =args .video .split (':')#line:978
            evalvideo (O000O0O000O00O0OO ,OO0000O0O00000000 ,OOO00OOO0O0O00OO0 )#line:979
        else :#line:980
            evalvideo (O000O0O000O00O0OO ,args .video )#line:981
        return #line:982
    OO000OO000O000OO0 =MovingAverage ()#line:984
    O00O0OOOO0O00O0O0 =len (O0000O000O00O0000 )if args .max_images <0 else min (args .max_images ,len (O0000O000O00O0000 ))#line:985
    O000O0O0OO0OO0O00 =ProgressBar (30 ,O00O0OOOO0O00O0O0 )#line:986
    print ()#line:988
    if not args .display and not args .benchmark :#line:990
        OOOOOOOOO0O000OOO ={'box':[[APDataObject ()for _O000OO0O000OO00O0 in cfg .dataset .class_names ]for _OOOO0O00OOO000OO0 in iou_thresholds ],'mask':[[APDataObject ()for _OO0OO00OO00000O0O in cfg .dataset .class_names ]for _O0OOO0OO0000O00O0 in iou_thresholds ]}#line:996
        O0OOO00OO0OO00000 =Detections ()#line:997
    else :#line:998
        timer .disable ('Load Data')#line:999
    O0O0O0O000O0O0OOO =list (range (len (O0000O000O00O0000 )))#line:1001
    if args .shuffle :#line:1003
        random .shuffle (O0O0O0O000O0O0OOO )#line:1004
    elif not args .no_sort :#line:1005
        O0O00OO0OOOO0O0O0 =[badhash (O00OO0OO0O0O000OO )for O00OO0OO0O0O000OO in O0000O000O00O0000 .ids ]#line:1014
        O0O0O0O000O0O0OOO .sort (key =lambda O0OO000000O0OOO0O :O0O00OO0OOOO0O0O0 [O0OO000000O0OOO0O ])#line:1015
    O0O0O0O000O0O0OOO =O0O0O0O000O0O0OOO [:O00O0OOOO0O00O0O0 ]#line:1017
    try :#line:1019
        for O0O0O0000OOO0O000 ,OOO0O0OOO00OOOO00 in enumerate (O0O0O0O000O0O0OOO ):#line:1021
            timer .reset ()#line:1022
            with timer .env ('Load Data'):#line:1024
                O0OO00000000000O0 ,O00OO000OO00OO0OO ,OOO0O0O0OO00OO00O ,O0000OO00O0OOO00O ,O0OO0OOOOO00O00OO ,OOO0O0O0OOOO00O0O =O0000O000O00O0000 .pull_item (OOO0O0OOO00OOOO00 )#line:1025
                if cfg .mask_proto_debug :#line:1028
                    with open ('scripts/info.txt','w')as OO00OOOO000OOO0O0 :#line:1029
                        OO00OOOO000OOO0O0 .write (str (O0000O000O00O0000 .ids [OOO0O0OOO00OOOO00 ]))#line:1030
                    np .save ('scripts/gt.npy',OOO0O0O0OO00OO00O )#line:1031
                O0O0O0000O0O0000O =Variable (O0OO00000000000O0 .unsqueeze (0 ))#line:1033
                if args .cuda :#line:1034
                    O0O0O0000O0O0000O =O0O0O0000O0O0000O .cuda ()#line:1035
            with timer .env ('Network Extra'):#line:1037
                OOO0000OO0OOO0OO0 =O000O0O000O00O0OO (O0O0O0000O0O0000O )#line:1038
            if args .display :#line:1040
                O0OO00OOOO000OOO0 =prep_display (OOO0000OO0OOO0OO0 ,O0OO00000000000O0 ,O0000OO00O0OOO00O ,O0OO0OOOOO00O00OO )#line:1041
            elif args .benchmark :#line:1042
                prep_benchmark (OOO0000OO0OOO0OO0 ,O0000OO00O0OOO00O ,O0OO0OOOOO00O00OO )#line:1043
            else :#line:1044
                prep_metrics (OOOOOOOOO0O000OOO ,OOO0000OO0OOO0OO0 ,O0OO00000000000O0 ,O00OO000OO00OO0OO ,OOO0O0O0OO00OO00O ,O0000OO00O0OOO00O ,O0OO0OOOOO00O00OO ,OOO0O0O0OOOO00O0O ,O0000O000O00O0000 .ids [OOO0O0OOO00OOOO00 ],O0OOO00OO0OO00000 )#line:1045
            if O0O0O0000OOO0O000 >1 :#line:1049
                OO000OO000O000OO0 .add (timer .total_time ())#line:1050
            if args .display :#line:1052
                if O0O0O0000OOO0O000 >1 :#line:1053
                    print ('Avg FPS: %.4f'%(1 /OO000OO000O000OO0 .get_avg ()))#line:1054
                plt .imshow (O0OO00OOOO000OOO0 )#line:1055
                plt .title (str (O0000O000O00O0000 .ids [OOO0O0OOO00OOOO00 ]))#line:1056
                plt .show ()#line:1057
            elif not args .no_bar :#line:1058
                if O0O0O0000OOO0O000 >1 :#line:1059
                    OO0O0OOO0OO00OOO0 =1 /OO000OO000O000OO0 .get_avg ()#line:1060
                else :#line:1061
                    OO0O0OOO0OO00OOO0 =0 #line:1062
                OO00OO0O0O0O0OO00 =(O0O0O0000OOO0O000 +1 )/O00O0OOOO0O00O0O0 *100 #line:1063
                O000O0O0OO0OO0O00 .set_val (O0O0O0000OOO0O000 +1 )#line:1064
                print ('\rProcessing Images  %s %6d / %6d (%5.2f%%)    %5.2f fps        '%(repr (O000O0O0OO0OO0O00 ),O0O0O0000OOO0O000 +1 ,O00O0OOOO0O00O0O0 ,OO00OO0O0O0O0OO00 ,OO0O0OOO0OO00OOO0 ),end ='')#line:1066
        if not args .display and not args .benchmark :#line:1068
            print ()#line:1069
            if args .output_coco_json :#line:1070
                print ('Dumping detections...')#line:1071
                if args .output_web_json :#line:1072
                    O0OOO00OO0OO00000 .dump_web ()#line:1073
                else :#line:1074
                    O0OOO00OO0OO00000 .dump ()#line:1075
            else :#line:1076
                if not train_mode :#line:1077
                    print ('Saving data...')#line:1078
                    with open (args .ap_data_file ,'wb')as OO00OOOO000OOO0O0 :#line:1079
                        pickle .dump (OOOOOOOOO0O000OOO ,OO00OOOO000OOO0O0 )#line:1080
                return calc_map (OOOOOOOOO0O000OOO )#line:1082
        elif args .benchmark :#line:1083
            print ()#line:1084
            print ()#line:1085
            print ('Stats for the last frame:')#line:1086
            timer .print_stats ()#line:1087
            O0000O00OOO000000 =OO000OO000O000OO0 .get_avg ()#line:1088
            print ('Average: %5.2f fps, %5.2f ms'%(1 /OO000OO000O000OO0 .get_avg (),1000 *O0000O00OOO000000 ))#line:1089
    except KeyboardInterrupt :#line:1091
        print ('Stopping...')#line:1092
def calc_map (O00O000O00O0OOOO0 ):#line:1095
    print ('Calculating mAP...')#line:1096
    OOO000OOO0O0OOO0O =[{'box':[],'mask':[]}for _O0OO00000OOOOO000 in iou_thresholds ]#line:1097
    for _O0O000000OOO00O00 in range (len (cfg .dataset .class_names )):#line:1099
        for OOO0000O0OOOOO00O in range (len (iou_thresholds )):#line:1100
            for OOO0O0000OOOO0OO0 in ('box','mask'):#line:1101
                O0O00000OOO0OO00O =O00O000O00O0OOOO0 [OOO0O0000OOOO0OO0 ][OOO0000O0OOOOO00O ][_O0O000000OOO00O00 ]#line:1102
                if not O0O00000OOO0OO00O .is_empty ():#line:1104
                    OOO000OOO0O0OOO0O [OOO0000O0OOOOO00O ][OOO0O0000OOOO0OO0 ].append (O0O00000OOO0OO00O .get_ap ())#line:1105
    OO0OO0000OOOOOOOO ={'box':OrderedDict (),'mask':OrderedDict ()}#line:1107
    for OOO0O0000OOOO0OO0 in ('box','mask'):#line:1110
        OO0OO0000OOOOOOOO [OOO0O0000OOOO0OO0 ]['all']=0 #line:1111
        for OO0O000OOO000OOOO ,O00OO0OOO00OOO0OO in enumerate (iou_thresholds ):#line:1112
            O00OO00O00O0O0000 =sum (OOO000OOO0O0OOO0O [OO0O000OOO000OOOO ][OOO0O0000OOOO0OO0 ])/len (OOO000OOO0O0OOO0O [OO0O000OOO000OOOO ][OOO0O0000OOOO0OO0 ])*100 if len (OOO000OOO0O0OOO0O [OO0O000OOO000OOOO ][OOO0O0000OOOO0OO0 ])>0 else 0 #line:1113
            OO0OO0000OOOOOOOO [OOO0O0000OOOO0OO0 ][int (O00OO0OOO00OOO0OO *100 )]=O00OO00O00O0O0000 #line:1114
        OO0OO0000OOOOOOOO [OOO0O0000OOOO0OO0 ]['all']=(sum (OO0OO0000OOOOOOOO [OOO0O0000OOOO0OO0 ].values ())/(len (OO0OO0000OOOOOOOO [OOO0O0000OOOO0OO0 ].values ())-1 ))#line:1115
    print_maps (OO0OO0000OOOOOOOO )#line:1117
    OO0OO0000OOOOOOOO ={O0O0O0O0O0OOOOOOO :{OO000OO0O00OOO00O :round (O00OO00OO00000O0O ,2 )for OO000OO0O00OOO00O ,O00OO00OO00000O0O in OO0OO00OOO0OOOOOO .items ()}for O0O0O0O0O0OOOOOOO ,OO0OO00OOO0OOOOOO in OO0OO0000OOOOOOOO .items ()}#line:1120
    return OO0OO0000OOOOOOOO #line:1121
def print_maps (OOO0OO0OO0O000OO0 ):#line:1124
    O0OO0O000O0OO00O0 =lambda O0O00OOO00OOOO00O :(' %5s |'*len (O0O00OOO00OOOO00O ))%tuple (O0O00OOO00OOOO00O )#line:1126
    OO0000O000OO00OOO =lambda O0O00OOOO0OO0O0O0 :('-------+'*O0O00OOOO0OO0O0O0 )#line:1127
    print ()#line:1129
    print (O0OO0O000O0OO00O0 (['']+[('.%d '%OOO000O00O00OO000 if isinstance (OOO000O00O00OO000 ,int )else OOO000O00O00OO000 +' ')for OOO000O00O00OO000 in OOO0OO0OO0O000OO0 ['box'].keys ()]))#line:1130
    print (OO0000O000OO00OOO (len (OOO0OO0OO0O000OO0 ['box'])+1 ))#line:1131
    for OOOOOOOOOOOOOO000 in ('box','mask'):#line:1132
        print (O0OO0O000O0OO00O0 ([OOOOOOOOOOOOOO000 ]+['%.2f'%O0OOOO00O0O0000O0 if O0OOOO00O0O0000O0 <100 else '%.1f'%O0OOOO00O0O0000O0 for O0OOOO00O0O0000O0 in OOO0OO0OO0O000OO0 [OOOOOOOOOOOOOO000 ].values ()]))#line:1133
    print (OO0000O000OO00OOO (len (OOO0OO0OO0O000OO0 ['box'])+1 ))#line:1134
    print ()#line:1135
if __name__ =='__main__':#line:1137
    parse_args ()#line:1138
    if args .config is not None :#line:1140
        set_cfg (args .config )#line:1141
    if args .trained_model =='interrupt':#line:1143
        args .trained_model =SavePath .get_interrupt ('weights/')#line:1144
    elif args .trained_model =='latest':#line:1145
        args .trained_model =SavePath .get_latest ('weights/',cfg .name )#line:1146
    if args .config is None :#line:1148
        model_path =SavePath .from_str (args .trained_model )#line:1149
        args .config =model_path .model_name +'_config'#line:1151
        print ('Config not specified. Parsed %s from the file name.\n'%args .config )#line:1152
        set_cfg (args .config )#line:1153
    if args .detect :#line:1155
        cfg .eval_mask_branch =False #line:1156
    if args .dataset is not None :#line:1158
        set_dataset (args .dataset )#line:1159
    with torch .no_grad ():#line:1161
        if not os .path .exists ('results'):#line:1162
            os .makedirs ('results')#line:1163
        if args .cuda :#line:1165
            cudnn .fastest =True #line:1166
            torch .set_default_tensor_type ('torch.cuda.FloatTensor')#line:1167
        else :#line:1168
            torch .set_default_tensor_type ('torch.FloatTensor')#line:1169
        if args .resume and not args .display :#line:1171
            with open (args .ap_data_file ,'rb')as f :#line:1172
                ap_data =pickle .load (f )#line:1173
            calc_map (ap_data )#line:1174
            exit ()#line:1175
        if args .image is None and args .video is None and args .images is None :#line:1177
            dataset =COCODetection (cfg .dataset .valid_images ,cfg .dataset .valid_info ,transform =BaseTransform (),has_gt =cfg .dataset .has_gt )#line:1179
            prep_coco_cats ()#line:1180
        else :#line:1181
            dataset =None #line:1182
        print ('Loading model...',end ='')#line:1184
        net =Yolact ()#line:1185
        net .load_weights (args .trained_model )#line:1186
        net .eval ()#line:1187
        print (' Done.')#line:1188
        if args .cuda :#line:1190
            net =net .cuda ()#line:1191
        evaluate_KDG (net ,dataset )#line:1194
