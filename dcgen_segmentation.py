from typing import Union
from PIL import Image, ImageDraw, ImageEnhance
import numpy as np
import json

from abc import ABC, abstractmethod

class ImgNode(ABC):
    # self.img: the image of the node
    # self.bbox: the bounding box of the node
    # self.children: the children of the node

    @abstractmethod
    def get_img(self):
        pass

class ImgSegmentation(ImgNode):
    def __init__(self, img: Union[str, Image.Image], bbox=None, children=None, max_depth=None, var_thresh=50, diff_thresh=45, diff_portion=0.9, window_size=50) -> None:
        if type(img) == str:
            img = Image.open(img)
        self.img = img
        # (left, top, right, bottom)
        self.bbox = (0, 0, img.size[0], img.size[1]) if not bbox else bbox
        self.children = children if children else []
        self.var_thresh = var_thresh
        self.diff_thresh = diff_thresh
        self.diff_portion = diff_portion
        self.window_size = window_size
        
        if max_depth:
            self.init_tree(max_depth)
        self.depth = self.get_depth()

    def init_tree(self, max_depth):
        def _init_tree(node, max_depth, cur_depth=0):
            if cur_depth == max_depth:
                return
            cuts = node.cut_img_bbox(node.img, node.bbox, line_direct="x")
            
            if len(cuts) == 0:
                cuts = node.cut_img_bbox(node.img, node.bbox, line_direct="y")

            # print(cuts)
            for cut in cuts:
                node.children.append(ImgSegmentation(node.img, cut, [], None, self.var_thresh, self.diff_thresh, self.diff_portion, self.window_size))

            for child in node.children:
                _init_tree(child, max_depth, cur_depth + 1)

        _init_tree(self, max_depth)

    def get_img(self, cut_out=False, outline=(0, 255, 0)):
        if cut_out:
            return self.img.crop(self.bbox)
        else:
            img_draw = self.img.copy()
            draw = ImageDraw.Draw(img_draw)
            draw.rectangle(self.bbox, outline=outline, width=5)
            return img_draw
    
    def display_tree(self, save_path=None):
        # draw a tree structure on the image, for each tree level, draw a different color
        def _display_tree(node, draw, color=(255, 0, 0), width=5):
            # deep copy the image
            draw.rectangle(node.bbox, outline=color, width=width)
            for child in node.children:
                # _display_tree(child, draw, color=tuple([int(random.random() * 255) for i in range(3)]), width=max(1, width))
                _display_tree(child, draw, color=color, width=max(1, width))

        img_draw = self.img.copy()
        draw = ImageDraw.Draw(img_draw)
        for child in self.children:
            _display_tree(child, draw)
        if save_path:
            img_draw.save(save_path)
        else:
            img_draw.show()

    def get_depth(self):
        def _get_depth(node):
            if node.children == []:
                return 1
            return 1 + max([_get_depth(child) for child in node.children])
        return _get_depth(self)
    
    def is_leaf(self):
        return self.children == []
    
    def to_json(self, path=None):
        '''
        [
            { "bbox": [left, top, right, bottom],
                "level": the level of the node,},
            { "bbox": [left, top, right, bottom],
            "level": the level of the node,}
            ...
        ]
        '''
        # use bfs to traverse the tree
        res = []
        queue = [(self, 0)]
        while queue:
            node, level = queue.pop(0)
            res.append({"bbox": node.bbox, "level": level})
            for child in node.children:
                queue.append((child, level + 1))
        if path:
            with open(path, "w") as f:
                json.dump(res, f, indent=4)
        return res
    
    def to_json_tree(self, path=None):
        '''
        {
            "bbox": [left, top, right, bottom],
            "children": [
                {
                    "bbox": [left, top, right, bottom],
                    "children": [ ... ]
                },
                ...
            ]
        }
        '''
        def _to_json_tree(node):
            res = {"bbox": node.bbox, "children": []}
            for child in node.children:
                res["children"].append(_to_json_tree(child))
            return res
        res = _to_json_tree(self)
        if path:
            with open(path, "w") as f:
                json.dump(res, f, indent=4)
        return res

    def cut_img_bbox(self, img, bbox,  line_direct="x", verbose=False, save_cut=False):
        """cut the the area of interest specified by bbox (left, top, right, bottom), return a list of bboxes of the cut image."""
        
        diff_thresh = self.diff_thresh
        diff_portion = self.diff_portion
        var_thresh = self.var_thresh
        sliding_window = self.window_size

        # def soft_separation_lines(img, bbox=None, var_thresh=None, diff_thresh=None, diff_portion=None, sliding_window=None):
        #     """return separation lines (relative to whole image) in the area of interest specified by bbox (left, top, right, bottom). 
        #     Good at identifying blanks and boarders, but not explicit lines. 
        #     Assume the image is already rotated if necessary, all lines are in x direction.
        #     Boundary lines are included."""
        #     img_array = np.array(img.convert("L"))
        #     img_array = img_array if bbox is None else img_array[bbox[1]:bbox[3]+1, bbox[0]:bbox[2]+1]
        #     offset = 0 if bbox is None else bbox[1]
        #     lines = []
        #     for i in range(1 + sliding_window, len(img_array) - 1):
        #         upper = img_array[i-sliding_window-1]
        #         window = img_array[i-sliding_window:i]
        #         lower = img_array[i]
        #         is_blank = np.var(window) < var_thresh
        #         # content width is larger than 33% of the width
        #         is_boarder_top = np.mean(np.abs(upper - window[0]) > diff_thresh) > diff_portion
        #         is_boarder_bottom = np.mean(np.abs(lower - window[-1]) > diff_thresh) > diff_portion
        #         if is_blank and (is_boarder_top or is_boarder_bottom):
        #             line = i if is_boarder_bottom else i - sliding_window
        #             lines.append(line + offset)
        #     return sorted(lines)
        def soft_separation_lines(img, bbox=None, var_thresh=None, diff_thresh=None, diff_portion=None, sliding_window=None):
            """return separation lines (relative to whole image) in the area of interest specified by bbox (left, top, right, bottom). 
            Good at identifying blanks and boarders, but not explicit lines. 
            Assume the image is already rotated if necessary, all lines are in x direction.
            Boundary lines are included."""
            img_array = np.array(img.convert("L"))
            img_array = img_array if bbox is None else img_array[bbox[1]:bbox[3]+1, bbox[0]:bbox[2]+1]
            # import matplotlib.pyplot as plt
            # # show the image array
            # plt.imshow(img_array, cmap="gray")
            # plt.show()

            offset = 0 if bbox is None else bbox[1]
            lines = []
            for i in range(2*sliding_window, len(img_array) - sliding_window):
                upper = img_array[i-2*sliding_window:i-sliding_window]
                window = img_array[i-sliding_window:i]
                lower = img_array[i:i+sliding_window]
                is_blank = np.var(window) < var_thresh
                # content width is larger than 33% of the width
                is_boarder_top = np.var(upper) > var_thresh
                is_boarder_bottom = np.var(lower) > var_thresh
                # print(i, "is_blank", is_blank, "is_boarder_top", is_boarder_top, "is_boarder_bottom", is_boarder_bottom)
                if is_blank and (is_boarder_top or is_boarder_bottom):
                    line = (i + i - sliding_window) // 2
                    lines.append(line + offset)

            # print(sorted(lines))
            return sorted(lines)

        def hard_separation_lines(img, bbox=None, var_thresh=None, diff_thresh=None, diff_portion=None):
            """return separation lines (relative to whole image) in the area of interest specified by bbox (left, top, right, bottom). 
            Good at identifying explicit lines (backgorund color change). 
            Assume the image is already rotated if necessary, all lines are in x direction
            Boundary lines are included."""
            img_array = np.array(img.convert("L"))
            # img.convert("L").show()
            img_array = img_array if bbox is None else img_array[bbox[1]:bbox[3]+1, bbox[0]:bbox[2]+1]
            offset = 0 if bbox is None else bbox[1]
            prev_row = None
            prev_row_idx = None
            lines = []

            # loop through the image array
            for i in range(len(img_array)):
                row = img_array[i]
                # if the row is too uniform, it's probably a line
                if np.var(img_array[i]) < var_thresh:
                    # print("row", i, "var", np.var(img_array[i]))
                    if prev_row is not None:
                        # the portion of two rows differ more that diff_thresh is larger than diff_portion
                        # print("prev_row", prev_row_idx, "diff", np.mean(np.abs(row - prev_row) > diff_thresh))
                        if np.mean(np.abs(row - prev_row) > diff_thresh) > diff_portion:
                            lines.append(i + offset)
                            # print("line", i)
                    prev_row = row
                    prev_row_idx = i
            # print(sorted(lines))
            return lines

        def new_bbox_after_rotate90(img, bbox, counterclockwise=True):
            """return the new coordinate of the bbox after rotating 90 degree, based on the original image."""
            if counterclockwise:
                # the top right corner of the original image becomes the origin of the coordinate after rotating 90 degree
                top_right = (img.size[0], 0)
                # change the origin
                bbox = (bbox[0] - top_right[0], bbox[1] - top_right[1], bbox[2] - top_right[0], bbox[3] - top_right[1])
                # rotate the bbox 90 degree counterclockwise (x direction change sign)
                bbox = (bbox[1], -bbox[2], bbox[3], -bbox[0])
            else:
                # the bottom left corner of the original image becomes the origin of the coordinate after rotating 90 degree
                bottom_left = (0, img.size[1])
                # change the origin
                bbox = (bbox[0] - bottom_left[0], bbox[1] - bottom_left[1], bbox[2] - bottom_left[0], bbox[3] - bottom_left[1])
                # rotate the bbox 90 degree clockwise (y direction change sign)
                bbox = (-bbox[3], bbox[0], -bbox[1], bbox[2])
            return bbox
        
        assert line_direct in ["x", "y"], "line_direct must be 'x' or 'y'"
        img = ImageEnhance.Sharpness(img).enhance(6)
        bbox = bbox if line_direct == "x" else new_bbox_after_rotate90(img, bbox, counterclockwise=True) # based on the original image
        img = img if line_direct == "x" else img.rotate(90, expand=True)
        lines = []
        # img.show()
        lines = soft_separation_lines(img, bbox, var_thresh, diff_thresh, diff_portion, sliding_window)
        lines += hard_separation_lines(img, bbox, var_thresh, diff_thresh, diff_portion)
        # print(hash(str(np.array(img).data)), bbox, var_thresh, diff_thresh, diff_portion, sliding_window, lines)
        if lines == []:
            return []
        lines = sorted(list(set([bbox[1],] + lines + [bbox[3],]))) # account for the beginning and the end of the image
        # list of images cut by the lines
        cut_imgs = []
        for i in range(1, len(lines)):
            cut = img.crop((bbox[0], lines[i-1], bbox[2], lines[i]))
            # if empty or too small, skip

            #! 주석처리 -> 작은것도 버리지 않음
            # if cut.size[1] < sliding_window:
            #     continue
            # elif np.array(cut.convert("L")).var() < var_thresh:
            #     continue
            cut = (bbox[0], lines[i-1], bbox[2], lines[i])  # (left, top, right, bottom)
            cut = cut if line_direct == "x" else new_bbox_after_rotate90(img, cut, counterclockwise=False)
            cut_imgs.append(cut)

        # if all other images are blank, this remaining image is the same as the original image
        if len(cut_imgs) == 1:
            return []
        if verbose:
            img = img if line_direct == "x" else img.rotate(-90, expand=True)
            draw = ImageDraw.Draw(img)
            for cut in cut_imgs:
                draw.rectangle(cut, outline=(0, 255, 0), width=5)
                draw.line(cut, fill=(0, 255, 0), width=5)
            img.show()
        if save_cut:
            img.save("cut.png")
        
        return cut_imgs