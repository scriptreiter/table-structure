import xlsxwriter
import os
import json

import dir_helper

def output(rows, cols, boxes, xlsx_path, json_path):
  try:
    os.remove(xlsx_path)
  except OSError:
    pass

  try:
    dir_helper.ensure(xlsx_path)
    book = xlsxwriter.Workbook(xlsx_path)
    sheet = book.add_worksheet()
  
    indices = {}
  
    for i, row in enumerate(rows):
  
      for box in row[5]:
        idx = boxes.index(box)
        indices[idx] = {}
  
        # This could be made more efficient with a default dict
        if 'rows' not in indices[idx]:
          indices[idx]['rows'] = []

        indices[idx]['rows'].append(i)
  
    for i, col in enumerate(cols):
      for box in col[5]:
        idx = boxes.index(box)

        # See note about efficiency above
        if 'cols' not in indices[idx]:
          indices[idx]['cols'] = []

        indices[idx]['cols'].append(i)

    cells = [[[] for x in range(len(cols))] for y in range(len(rows))]

    for i,box in enumerate(boxes):
      sorted_rows = sorted(indices[i]['rows'])
      sorted_cols = sorted(indices[i]['cols'])
      for k,row_idx in enumerate(sorted_rows):
        for j,col_idx in enumerate(sorted_cols):
          if k == 0 and j == 0:
            contents = {'type': 'cell', 'contents': box}
          else:
            contents = {'type': 'span', 'main_row': sorted_rows[0], 'main_col': sorted_cols[0]}

          cells[row_idx][col_idx].append(contents)

    out_arr = [['' for x in range(len(cols))] for y in range(len(rows))]

    for row_idx, row in enumerate(cells):
      for col_idx, cell in enumerate(row):
        overall = []
        for cell_info in cell:
          if 'type' not in cell_info:
            cell_info = {'type': 'unspecified'}

          if cell_info['type'] == 'cell':
            contents = ' '.join(cell_info['contents'][4])
          elif cell_info['type'] == 'span':
            # This can later be a special structure for the json
            # including for the main cell, too
            contents = 'SPAN_OF(' + str(cell_info['main_row']) + ', ' + str(cell_info['main_col']) + ')'
          else:
            contents = ''

          overall.append(contents)

        display_contents = ' '.join(overall)
        sheet.write(row_idx, col_idx, display_contents)

        # Store for json output
        out_arr[row_idx][col_idx] = display_contents

    dir_helper.ensure(json_path)
    with open(json_path, 'w') as f:
      json.dump({'cells': out_arr}, f)
  
    # sheet.write(row, col, item)

  finally:
    book.close()
