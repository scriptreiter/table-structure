import numpy as np
import cv2
from itertools import combinations

verbose = False

def get_lines(img_name, base_path):
  img = cv2.imread(base_path + '/' + img_name)
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  edges = cv2.Canny(gray, 50, 150, apertureSize = 3)
  # cv2.imwrite('regents/canny/' + img_name, edges)
  # 120, 20, 10 is good. Also 80, 20, 1
  lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 120, minLineLength=40, maxLineGap=2)

  if lines is None:
    lines = []
  
  horiz_count = 0
  vert_count = 0
  
  horiz_lines = []
  vert_lines = []
  
  for info in lines:
    x1, y1, x2, y2 = info[0]

    line_info = {}

    if abs(y1 - y2) < 0.1:
      # This is a horizontal line
      line_info['border'] = int((y1 + y2) / 2)
      line_info['start'] = x1
      line_info['end'] = x2
      horiz_lines.append(line_info)
      horiz_count += 1
    elif abs(x1 - x2) < 0.1:
      # This is a vertical line
      line_info['border'] = int((x1 + x2) / 2)
      line_info['start'] = y1
      line_info['end'] = y2
      vert_lines.append(line_info)
      vert_count += 1
    elif verbose:
      print('Nonstandard line: ' + str(theta))

  return (horiz_lines, vert_lines)
  
def get_sorted_avg_lines(lines):
  horiz_lines = lines[0]
  vert_lines = lines[1]

  # Now bin the lines
  tolerance = 6
  horiz_bins = []
  vert_bins = []
  
  for line in horiz_lines:
    bin_value(line, horiz_bins, tolerance)
  
  for line in vert_lines:
    bin_value(line, vert_bins, tolerance)
  
  # Now average out the bins
  horiz_markers = average_bins(horiz_bins)
  horiz_markers.sort()
  
  vert_markers = average_bins(vert_bins)
  vert_markers.sort()

  return (horiz_markers, vert_markers)

# Could remove the negative scoring lines beforehand...
# Or change the scores of the filtered lines to be negative
def remove_lines(lines, filtered_lines, scores):
  new_horiz_lines = cut_lines(lines[0], filtered_lines[0], scores[0])
  new_vert_lines = cut_lines(lines[1], filtered_lines[1], scores[1])

  return (new_horiz_lines, new_vert_lines)

def cut_lines(lines, filtered, scores):
  new_lines = []
  for i, line in enumerate(lines):
    # Could do this more efficiently, particularly since it's sorted
    if i not in filtered and scores[i] > 0:
      new_lines.append(line)

  return new_lines

def filter_lines(lines, boxes, scores):
  horiz_lines = lines[0]
  vert_lines = lines[1]

  horiz_scores = scores[0]
  vert_scores = scores[1]

  horiz_removed_lines = check_lines(horiz_lines, boxes, horiz_scores, 1)
  vert_removed_lines = check_lines(vert_lines, boxes, vert_scores, 0)

  if verbose:
    for line_i in horiz_removed_lines:
      print('Removed h line #' + str(line_i) + ' at ' + str(horiz_lines[line_i]));

    for line_i in vert_removed_lines:
      print('Removed v line #' + str(line_i) + ' at ' + str(vert_lines[line_i]));

  return (horiz_removed_lines, vert_removed_lines)

def check_lines(lines, boxes, scores, offset):
  removed_lines = set()
  for comb in combinations(enumerate(lines), 2):
    line_1 = comb[0][1]
    line_2 = comb[1][1]
    min_val = min(line_1['border'], line_2['border'])
    max_val = max(line_1['border'], line_2['border'])

    box_inbetween = False
    for box in boxes:
      box_edge_1 = box[offset]
      box_edge_2 = box[offset] + box[offset + 2]
      if (box_edge_1 < max_val and box_edge_1 > min_val) or (box_edge_2 < max_val and box_edge_2 > min_val):
        box_inbetween = True

    # Quick hack for now to avoid cutting two line segments at the same
    # offset. If there is no box in between, then we can just check if they are on the
    # same offset. We may do something more elegant later
    if not box_inbetween and line_1['border'] != line_2['border']:
      # We need to choose one line over the other
      if scores[comb[0][0]] > scores[comb[1][0]]:
        removed_lines.add(comb[1][0])
      else:
        removed_lines.add(comb[0][0])

  return removed_lines

def rate_lines(lines, boxes):
  horiz_lines = lines[0]
  vert_lines = lines[1]
  horiz_lines.sort(key = lambda info: info['border'])
  vert_lines.sort(key = lambda info: info['border'])

  horiz_scores = calc_line_box_scores(horiz_lines, boxes, 1)
  vert_scores = calc_line_box_scores(vert_lines, boxes, 0)

  if verbose:
    for i, line in enumerate(horiz_lines):
      print('H Line at ' + str(line) + ' with a score of ' + str(horiz_scores[i]))

    for i, line in enumerate(vert_lines):
      print('V Line at ' + str(line) + ' with a score of ' + str(vert_scores[i]))

  return (horiz_scores, vert_scores)

def calc_line_box_scores(lines, boxes, offset):
  scores = []

  for line in lines:
    min_margin = float('inf')

    num_intersections = 0
    intersection_penalty = 0

    for box in boxes:
      first_edge = box[offset]
      second_edge = box[offset] + box[offset + 2]

      alt_offset = (offset + 1) % 2
      first_alt_edge = box[alt_offset]
      second_alt_edge = box[alt_offset + 2]


      # We only want to consider the box if the line overlaps it somewhat
      overlap = max(0, min(line['end'], second_alt_edge) - max(line['start'], first_alt_edge))

      if overlap > 0:
        # Calculate the minimum distance to either edge of this box
        min_to_edge = min(abs(line['border'] - first_edge), abs(line['border'] - second_edge))
  
        # If it intersects the box, track that, and penalize based
        # on how far into the box it is
        if line['border'] >= first_edge and line['border'] <= second_edge:
          num_intersections += 1
          intersection_penalty += min_to_edge
  
        # Could track some sense of uniformity in the closest edges
        # Although the case of this provides a problem:
        #
        # Line 1
        # Line 2        Line 1      Line 1
        # Line 3
        #
        # Because it would preference grid lines through cell 1
  
        # Check if this is the smallest margin, yet
        if min_to_edge < min_margin:
          min_margin = min_to_edge

    score = min_margin - (num_intersections * intersection_penalty)

    scores.append(score)

  return scores

def bin_value(line, bins, tolerance):
  for bin in bins:
    for bin_line in bin:
      if abs(line['border'] - bin_line['border']) < tolerance:
        bin.append(line)
        return

  # Not within tolerance of any bin
  bins.append([line])

def display_bins(bins):
  if verbose:
    for bin in bins:
      print(bin)

def average_bins(bins):
  averaged_bins = []
  for bin in bins:
    averaged_bins.append(sum(bin) / len(bin))

  return averaged_bins

