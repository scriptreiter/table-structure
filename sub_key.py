def get_key():
  with open ('../../config/ocr.key', 'r') as ocr_file:
    ocr_key = ocr_file.read().rstrip()

  return ocr_key
