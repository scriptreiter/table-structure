import numpy as np
import cv2
from itertools import combinations

def get_contours(img_name, base_path):
  img = cv2.imread(base_path + '/' + img_name)
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  bin_img = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)[1]

  (edges, contours, hierarchy) = cv2.findContours(bin_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

  rects = []
  approxes = []

  for i,contour in enumerate(contours):
    perim = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, perim * 0.02, True)
    approxes.append(approx)

    # If approximated with a quadrilateral, we want to save
    # and cv2.contourArea(cnt) > 1000 and cv2.isContourConvex(cnt)
    # Going to need to check if it detects rectangles with multiple cells, and cut those elsewhere
    # and use them for table label info like title, etc.

    if len(approx) == 4 and cv2.isContourConvex(approx) and cv2.contourArea(approx) > 200:
      temp_contour = approx.reshape(-1, 2)
      max_cos = np.max([angle_cos(temp_contour[i], temp_contour[(i+1) % 4], temp_contour[(i+2) % 4]) for i in range(4)])

      if max_cos < 0.1:
        rects.append((i,approx)) # Keep the index and the contour

  return (rects, hierarchy)

def get_rects(idxs, rects):
  matched = []
  flattened = {}

  for idx,rect in rects:
    flattened[idx] = rect

  for idx in idxs:
    matched.append((idx, flattened[idx]))

  return matched

def get_most_nested(contours, hierarchy, rects):
  if len(contours) < 1:
    return None

  max_total = 0
  max_contour = contours[0]
  max_children = []

  for contour in contours:
    children = get_all_children(contour[0], hierarchy, rects)
    total_children = len(children)
    
    if total_children > max_total:
      max_total = total_children
      max_contour = contour
      max_children = children

  return (max_contour, max_children)

def get_child_contours(rects, hierarchy):
  idxes = [i for (i, c) in rects]

  # Select the contours in this group that are child contours
  # These should be cell-level contours, hopefully
  return [contour for (idx, contour) in rects if no_children(idxes, hierarchy, idx)]

def get_all_children(idx, hierarchy, rects):
  children = []
  new_children = get_children(hierarchy, idx)

  while len(new_children) > 0:
    children += new_children
    grandchildren = []

    for i in new_children:
      grandchildren += get_children(hierarchy, i)

    new_children = grandchildren

  return [x[0] for x in rects if x[0] in children]

def no_children(idxes, hierarchy, idx):
  children = get_children(hierarchy, idx)

  while len(children) > 0:
    curr_idx = children.pop()

    # If the current child is in the list
    # Then we can shortcut and exit
    if curr_idx in idxes:
      return False

    # Get any children of the current child
    children += get_children(hierarchy, curr_idx)

  return True

def get_children(hierarchy, idx):
  return [i for i,x in enumerate(hierarchy[0]) if x[3] == idx]

def get_root_contours(rects, hierarchy):

  if len(rects) == 0:
    return []

  # Select the contour in this group with no parents
  # Hopefully there is only one, and this is the whole image
  parent_idxes = [idx for (idx, contour) in rects if hierarchy[0][idx][3] == -1]

  if len(parent_idxes) < 1:
    return []

  parent_idx = parent_idxes[0]

  # Now want to select the ones that are a direct child of this one
  # Biggest is likely the table
  root_rects = [(idx, contour) for (idx, contour) in rects if hierarchy[0][idx][3] == parent_idx]

  return root_rects

def get_largest_contour(contours):
  max_area = 0.0
  max_contour = contours[0]

  for i,contour in contours:
    area = cv2.contourArea(contour)

    if area > max_area:
      max_area = area
      max_contour = contour

  return max_contour

def angle_cos(p0, p1, p2):
  d1, d2 = (p0-p1).astype('float'), (p2-p1).astype('float')
  return abs(np.dot(d1, d2) / np.sqrt(np.dot(d1, d1)*np.dot(d2, d2)))

def contour_to_box(contour):
  max_x = max_y = float('-inf')
  min_x = min_y = float('inf')

  for point in contour:
    max_x = max(max_x, point[0][0])
    max_y = max(max_y, point[0][1])
    min_x = min(min_x, point[0][0])
    min_y = min(min_y, point[0][1])

  # Storing as x, y, width, height)
  return (min_x, min_y, max_x - min_x, max_y - min_y, '')
def contours_to_boxes(contours):
  boxes = []
  for contour in contours:
    boxes.append(contour_to_box(contour))

  return boxes
