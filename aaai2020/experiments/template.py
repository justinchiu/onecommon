from jinja2 import Template

utterance_template = Template("{{ confirm }} {{ mention }}")

confirm = "Yes ."
disconfirm = "No ."

# mention 1
# Do you see a small grey dot?
# mention 2
# Do you see a pair of dots, where the bottom left one is small and grey and the top right one is medium and light?
# mention 3
# Do you see three dots, where the {{dot 1}}, {{dot 2}}, and {{dot 3}}?

mention_1 = Template("a {{dot1}}")
mention_2 = Template("a pair of dots, with the {{dot1}} and {{dot2}}")
mention_3 = Template("three dots, {{dot1}}, {{dot2}}, {{dot3}}")

# selection
select_1 = Template("")


import numpy as np
from belief import process_ctx, OrBelief

class ConfigFeatures:
    def __init__(self, num_dots, sc, xy):
        self.num_dots = num_dots
        self.sc = sc
        self.xy = xy


    def set_dots(self):
        self.dots = [
            Dot(
                size = self.sc[i,0],
                color = self.sc[i, 1],
                xy = self.xy[i],
            ) for i in range(self.num_dots)
        ]

num_dots = 7
ctx = np.array([
    0.635, -0.4,   2/3, -1/6,  # 8
    0.395, -0.7,   0.0,  3/4,  # 11
    -0.74,  0.09,  2/3, -2/3,  # 13
    -0.24, -0.63, -1/3, -1/6,  # 15
    0.15,  -0.58,  0.0,  0.24, # 40
    -0.295, 0.685, 0.0, -8/9,  # 50
    0.035, -0.79, -2/3,  0.56, # 77
], dtype=float).reshape(-1, 4)
ids = np.array([8, 11, 13, 15, 40, 50, 77], dtype=int)

# goes in init
size_color = process_ctx(ctx)
xy = ctx[:,:2]

utt = np.array([1,0,1,1,0,0,0])
utt = np.array([1,0,1,1,0,1,0])

belief = OrBelief(num_dots, ctx, absolute=True)
utt_feats = belief.get_feats(utt)
matches = belief.resolve_utt(*utt_feats)

# generate utterance for particular dot in triangle
# num dots: int, size_color: n x 2, xy positions: n x 2
n, sc, xy = utt_feats

def is_only(dots, idx):
    return dots[idx] and dots.sum() == 1

def centroid(xy):
    right, top = xy.max(0)
    left, bottom = xy.min(0)
    return (right + left) / 2, (top + bottom) / 2

def relative_position(x, y, mx, my):
    if x < mx and y > my:
        return "top left"
    elif x > mx and y > my:
        return "top right"
    elif x < mx and y < my:
        return "bottom left"
    elif x > mx and y < my:
        return "bottom right"
    else:
        raise ValueError

def spatial_descriptions2(xy):
    assert xy.shape[0] == 2
    right, top = xy.max(0)
    left, bottom = xy.min(0)
    mx, my = (right + left) / 2, (top + bottom) / 2

    radius = .1
    horizontal_close = abs(top - bottom) < 2 * radius
    vertical_close = abs(right - left) < 2 * radius

    if horizontal_close and not vertical_close:
        # check if dots are close horizontally
        return [
            "top" if xy[0,1] > my else "bottom",
            "top" if xy[1,1] > my else "bottom",
        ]
    elif vertical_close and not horizontal_close:
        # check if dots are close vertically
        return [
            "left" if xy[0,0] < mx else "right",
            "left" if xy[1,0] < mx else "right",
        ]
    else:
        # otherwise use full description
        return [
            relative_position(xy[0,0], xy[0,1], mx, my),
            relative_position(xy[1,0], xy[1,1], mx, my),
        ]


def spatial_descriptions3(xy):
    assert xy.shape[0] == 3
    right, top = xy.max(0)
    left, bottom = xy.min(0)
    mx, my = (right + left) / 2, (top + bottom) / 2

    top_dots = xy[:,1] > my
    bottom_dots = xy[:,1] < my

    right_dots = xy[:,0] > mx
    left_dots = xy[:,0] < mx

    # possible configurations:
    # * full rank triangle
    # * low rank line
    # * single dot? not likely
    descriptions = []
    for idx in range(xy.shape[0]):
        is_top = top_dots[idx]
        is_bottom = bottom_dots[idx]
        is_left = left_dots[idx]
        is_right = right_dots[idx]

        if is_only(top_dots, idx):
            descriptions.append("top")
        elif is_only(bottom_dots, idx):
            descriptions.append("bottom")
        elif is_only(left_dots, idx):
            descriptions.append("left")
        elif is_only(right_dots, idx):
            descriptions.append("right")
        elif is_top and is_left:
            descriptions.append("top left")
        elif is_top and is_right:
            descriptions.append("top right")
        elif is_bottom and is_left:
            descriptions.append("bottom left")
        elif is_bottom and is_right:
            descriptions.append("bottom right")
        else:
            raise ValueError
    return descriptions

def spatial_descriptions4(xy):
    assert xy.shape[0] == 4
    right, top = xy.max(0)
    left, bottom = xy.min(0)
    mx, my = (right + left) / 2, (top + bottom) / 2

    top_dots = xy[:,1] > my
    bottom_dots = xy[:,1] < my

    right_dots = xy[:,0] > mx
    left_dots = xy[:,0] < mx

    # possible configurations:
    # * full rank quadrilateral
    # * low rank triangle
    # * low rank line
    # * single dot
    descriptions = []
    for idx in range(xy.shape[0]):
        is_top = top_dots[idx]
        is_bottom = bottom_dots[idx]
        is_left = left_dots[idx]
        is_right = right_dots[idx]

        if is_only(top_dots, idx):
            descriptions.append("top")
        elif is_only(bottom_dots, idx):
            descriptions.append("bottom")
        elif is_only(left_dots, idx):
            descriptions.append("left")
        elif is_only(right_dots, idx):
            descriptions.append("right")
        elif is_top and is_left:
            descriptions.append("top left")
        elif is_top and is_right:
            descriptions.append("top right")
        elif is_bottom and is_left:
            descriptions.append("bottom left")
        elif is_bottom and is_right:
            descriptions.append("bottom right")
        else:
            raise ValueError
    return descriptions

xy3_descriptions = spatial_descriptions3(xy[:3])
xy2_descriptions = spatial_descriptions2(xy[:2])

def size_color_descriptions(sc):
    size_map = ["small", "medium", "large"]
    color_map = ["light", "grey", "dark"]
    return [
        (size_map[x[0]], color_map[x[1]]) for x in sc
    ]

import streamlit as st

print(xy3_descriptions)
print(xy2_descriptions)

print(sc)
print(size_color_descriptions(sc))

import matplotlib.pyplot as plt
# sc = {0,1,2}
plt.scatter(xy[:,0],xy[:,1], s=(sc[:,0] + 1) * 10, c=(sc[:,1] + 1) * 5)
plt.show()

import pdb; pdb.set_trace()
