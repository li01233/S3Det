import numpy as np
import math
def slice_image(image_origin,annotation_origin,cls_origin,
                slice_height,slice_width,
                overlap_height_ratio: float = 0.2,
                overlap_width_ratio: float = 0.2,small_object_theshold=0.001):
    # save every img roi
    image_list = []
    label_dict = dict()
    # save all slice's anno and cls
    cls_dict = {}
    anno_dict = {}

    small_flag = False

    h_origin,w_origin  = image_origin
    box_origin = []
    
    # change YOLO format annotations to origin [xywh]
    for i in range(len(annotation_origin)):
        box = annotation_origin[i]
        cls = cls_origin[i]

        x = box[0] * w_origin
        y = box[1] * h_origin
        w = box[2] * w_origin
        h = box[3] * h_origin
        
        if (w*h) / (w_origin*h_origin) < small_object_theshold:
            small_flag = True

        x1 = x - w/2
        x2 = x + w/2
        y1 = y - h/2
        y2 = y + h/2

        box_origin.append([cls,x1,y1,x2,y2])
    
    # judge if there are small objects
    if small_flag is not True:
        anno_dict[(0,0,w_origin,h_origin)] = annotation_origin
        label_dict['bboxes'] = anno_dict
        cls_dict[(0,0,w_origin,h_origin)] = cls_origin
        label_dict['cls'] = cls_dict
        return label_dict
    
    # if slice is larger
    slice_width = min(slice_width,w_origin)
    slice_height = min(slice_height,h_origin)
    
    # slice img and anno
    overlap_width = int(overlap_width_ratio * slice_width)
    overlap_height = int(overlap_height_ratio * slice_height)
    start_x = 0
    start_y = 0 
    anno_list = [] # save cal result
    cls_list = []
    for _ in range(math.ceil((h_origin-overlap_height)/(slice_height-overlap_height))):
        for _ in range(math.ceil((w_origin-overlap_width)/(slice_width-overlap_width))):
            # use roi to get img
            end_x = start_x + slice_width
            end_y = start_y + slice_height
            if (end_x > w_origin) or (end_y > h_origin):
                if(start_x < w_origin) and (start_y < h_origin):
                    end_x = min(end_x,w_origin)
                    end_y = min(end_y,h_origin)
                else:
                    break
                        
            # calculate slicing anno
            if_box_flag = False
            anno_tmp = []
            cls_tmp = []
            for box in box_origin:
                if (box[1] < end_x) and (box[2] < end_y) and (box[3] > start_x) and (box[4] > start_y):
                    x1 = max(box[1],start_x)-start_x
                    y1 = max(box[2],start_y)-start_y
                    x2 = min(box[3],end_x)-start_x
                    y2 = min(box[4],end_y)-start_y
                    if_box_flag = True
                     
                    anno_tmp.append([x1,y1,x2,y2])
                    cls_tmp.append(cls)

            # if this slice have box then add it to list
            if if_box_flag:
                anno_list.append(anno_tmp) #[slice1:[box1:[],box2:[]],slice2:[]]
                cls_list.append(cls_tmp)
                image_list.append((start_x,start_y,end_x,end_y)) 
                anno_tmp = []

            start_x = start_x + slice_width - overlap_width
        
        start_x = 0
        start_y = start_y + slice_height - overlap_height

    # change origin [xywh] to YOLO format annotations
    anno_tmp = []
    for i in range(len(image_list)):
        roi = image_list[i]
        slice = anno_list[i]
        cls_tmp = cls_list[i]

        for anno in slice:
            x = ((anno[0]+anno[2])/2)/slice_width
            y = ((anno[1]+anno[3])/2)/slice_height
            w = max(anno[2]-anno[0],0)/slice_width
            h = max(anno[3]-anno[1],0)/slice_height
            anno_tmp.append([x,y,w,h])

        anno_dict[roi] = np.array(anno_tmp)
        cls_dict[roi] = np.array(cls_tmp)
        anno_tmp = []
    
    label_dict['bboxes'] = anno_dict
    label_dict['cls'] = cls_dict
    return label_dict