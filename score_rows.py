from itertools import combinations

import clusterer

def get_structure(boxes, lines):
  row_clusters, col_clusters = rate_combinations(boxes, lines)

  rows = translate_clusters(row_clusters)
  cols = translate_clusters(col_clusters)

  sorted_rows = sorted(rows, key = lambda row: (row[1], row[7]))
  sorted_cols = sorted(cols, key = lambda col: (col[0], col[6]))

  return (sorted_rows, sorted_cols)

def combine_boxes(box1, box2):
  return (
    min(box1[0], box2[0]),
    min(box1[1], box2[1]),
    max(box1[2], box2[2]),
    max(box1[3], box2[3]),
    box1[4] + box2[4],
    box1[5] + box2[5]
  )

def translate_clusters(clusters):
  combined = []
  for cluster in clusters:
    labels = []
    boxes = []
    max_x = max_y = max_min_x = max_min_y = float("-inf")
    min_x = min_y = float("inf")

    for box in cluster:
      labels += box[4]
      boxes.append(box)
      min_x = min(min_x, box[0])
      max_x = max(max_x, box[0] + box[2])
      min_y = min(min_y, box[1])
      max_y = max(max_y, box[1] + box[3])

      # To allow sorting of two rows joined by a span (who share
      # the same min_x, we want to record the maximum min_x of a box
      # to allow secondary sorting on this. The same applies
      # with y for rows

      max_min_x = max(max_min_x, box[0])
      max_min_y = max(max_min_y, box[1])

    combined.append((min_x, min_y, max_x, max_y, labels, boxes, max_min_x, max_min_y))

  return combined

def rate_combinations(boxes, lines):
  overall_row_scores = {}
  row_score_matrix = [[1.0 for x in range(len(boxes))] for y in range(len(boxes))]
  overall_col_scores = {}
  col_score_matrix = [[1.0 for x in range(len(boxes))] for y in range(len(boxes))]
  horiz_lines = lines[0]
  vert_lines = lines[1]

  for comb in combinations(enumerate(boxes), 2):
    row_scores = {}
    col_scores = {}

    i = comb[0][0]
    j = comb[1][0]

    box_1 = {
      'left': comb[0][1][0],
      'right': comb[0][1][0] + comb[0][1][2],
      'top': comb[0][1][1],
      'bottom': comb[0][1][1] + comb[0][1][3]
    }

    box_2 = {
      'left': comb[1][1][0],
      'right': comb[1][1][0] + comb[1][1][2],
      'top': comb[1][1][1],
      'bottom': comb[1][1][1] + comb[1][1][3]
    }

    # 1.) Their vertical (horizontal) centers align
    # May want to cut the factor down to 1.0 to make it a max of 1.0
    row_scores['center_align'] = 2.0 / (1.0 + abs(box_1['top'] + box_1['bottom'] - box_2['top'] - box_2['bottom']))
    col_scores['center_align'] = 2.0 / (1.0 + abs(box_1['left'] + box_1['right'] - box_2['left'] - box_2['right']))

    # 2.) Their left (top) edges align
    row_scores['top_align'] = 1.0 / (1.0 + abs(box_1['top'] - box_2['top']))
    col_scores['left_align'] = 1.0 / (1.0 + abs(box_1['left'] - box_2['left']))

    # 3.) Their right (bottom) edges align
    row_scores['bottom_align'] = 1.0 / (1.0 + abs(box_1['bottom'] - box_2['bottom']))
    col_scores['right_align'] = 1.0 / (1.0 + abs(box_1['right'] - box_2['right']))

    # 4.) If there is a line close to their left (above them)
    row_scores['top_line'] = calculate_preceding_line_score(box_1['top'], box_2['top'], horiz_lines)
    col_scores['left_line'] = calculate_preceding_line_score(box_1['left'], box_2['left'], vert_lines)

    # 5.) If there is a line close to their right (below them)
    row_scores['bottom_line'] = calculate_succeeding_line_score(box_1['bottom'], box_2['bottom'], horiz_lines)
    col_scores['right_line'] = calculate_succeeding_line_score(box_1['right'], box_2['right'], vert_lines)

    # 6.) They overlap significantly in their horizontal (vertical) range
    row_scores['vert_overlap'] = calculate_vertical_overlap(box_1, box_2)
    col_scores['horiz_overlap'] = calculate_horizontal_overlap(box_1, box_2)

    # 7.) I would like to add in a term regarding a shared strong score with a third object

    row_score = calculate_row_score(row_scores)
    col_score = calculate_col_score(col_scores)

    overall_row_scores[str(comb)] = row_score
    overall_col_scores[str(comb)] = col_score

    row_score_matrix[comb[0][0]][comb[1][0]] = row_score
    row_score_matrix[comb[1][0]][comb[0][0]] = row_score
    col_score_matrix[comb[0][0]][comb[1][0]] = col_score
    col_score_matrix[comb[1][0]][comb[0][0]] = col_score

  # Might want to do 0.999 later
  row_clusters = clusterer.newer_cluster_scores(row_score_matrix, 1.0)
  col_clusters = clusterer.newer_cluster_scores(col_score_matrix, 1.0)

  # Now translate the clusters of indexes into clusters of boxes

  row_cluster_boxes = []

  for row in row_clusters:
    row_cluster_boxes.append([])
    for box_index in row:
      row_cluster_boxes[len(row_cluster_boxes) - 1].append(boxes[box_index])

  col_cluster_boxes = []
  for col in col_clusters:
    col_cluster_boxes.append([])
    for box_index in col:
      col_cluster_boxes[len(col_cluster_boxes) - 1].append(boxes[box_index])

  return (row_cluster_boxes, col_cluster_boxes)

# We may eventually want to take into account the proximity of a line, as this
# will currently treat a line as infinte, ignoring our endpoints. I think
# this is okay behavior, as the presence of a line close above (left) still suggests
# row (col) structure even if offset somewhat
def calculate_preceding_line_score(edge1, edge2, lines):
  min_edge = min(edge1, edge2)
  min_dist = float('inf')
  min_line = {'border': 0}

  for line in lines:
    if line['border'] <= min_edge and min_edge - line['border'] < min_dist:
      min_dist = min_edge - line['border']
      min_line = line

  # Could also just consider only the minimum, instead of both, but
  # I think both is probably better, for example in the case of two
  # With one far away from the other
  return 1.0 / (1.0 + edge1 - min_line['border'] + edge2 - min_line['border'])

# See the note above (for calculate_preceding_line_score) regarding
# doing this for segmented lines, possibly
def calculate_succeeding_line_score(edge1, edge2, lines):
  max_edge = max(edge1, edge2)
  min_dist = float('inf')
  min_line = {'border': 0}

  for line in lines:
    if line['border'] >= max_edge and max_edge - line['border'] < min_dist:
      min_dist = max_edge - line['border']
      min_line = line

  # Could also just consider only the maximum, instead of both, but
  # I think both is probably better, for example in the case of two
  # With one far away from the other
  return 1.0 / (1.0 + min_line['border'] - edge1 + min_line['border'] - edge2)

# Returns percent overlap (length of overlap over min side interval length)
def calculate_vertical_overlap(box1, box2):
  inter_len = max(0, min(box1['bottom'], box2['bottom']) - max(box1['top'], box2['top']))
  min_size = min(box1['bottom'] - box1['top'], box2['bottom'] - box2['top'])

  return inter_len * 1.0 / min_size

# Returns percent overlap (length of overlap over min side interval length)
def calculate_horizontal_overlap(box1, box2):
  inter_len = max(0, min(box1['right'], box2['right']) - max(box1['left'], box2['left']))
  min_size = min(box1['right'] - box1['left'], box2['right'] - box2['left'])

  return inter_len * 1.0 / min_size

def calculate_row_score(scores):
  score = 0.0

  for score_type in scores:
    score += scores[score_type]

  return score

def calculate_col_score(scores):
  return calculate_row_score(scores) # Same for now
