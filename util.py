import easyocr
import string
import cv2
from PIL import Image, ImageDraw, ImageFont

dict_char_to_int = {'O': '0',
                    'I': '1',
                    'J': '3',
                    'A': '4',
                    'G': '6',
                    'S': '5',
                    'B': '8'}

dict_int_to_char = {'0': 'O',
                    '1': 'I',
                    '3': 'J',
                    '4': 'A',
                    '6': 'G',
                    '5': 'S',
                    '8': 'B'}

reader = easyocr.Reader(['en'])

def get_car(plate_bbox, lst_car_results):
    plate_x1, plate_y1, plate_x2, plate_y2 = plate_bbox
    foundIt = False
    for i in range(len(lst_car_results)):
        car_x1, car_y1, car_x2, car_y2, _, _ = lst_car_results[i]

        if plate_x1 > car_x1 and  plate_y1 > car_y1 and  plate_x2 < car_x2 and plate_y2 < car_y2:
            car_indx = i
            foundIt = True
            break

    if foundIt:
        return lst_car_results[car_indx]

    return -1, -1, -1, -1, -1, -1

def license_complies_format(text):
    if len(text) != 7:
        return False

    if (text[0] in string.ascii_uppercase or text[0] in dict_int_to_char.keys()) and \
       (text[1] in string.ascii_uppercase or text[1] in dict_int_to_char.keys()) and \
       (text[2] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[2] in dict_char_to_int.keys()) and \
       (text[3] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[3] in dict_char_to_int.keys()) and \
       (text[4] in string.ascii_uppercase or text[4] in dict_int_to_char.keys()) and \
       (text[5] in string.ascii_uppercase or text[5] in dict_int_to_char.keys()) and \
       (text[6] in string.ascii_uppercase or text[6] in dict_int_to_char.keys()):
        return True
    else:
        return False
    
def format_license(text):
    license_plate_ = ''
    mapping = {0: dict_int_to_char, 1: dict_int_to_char, 4: dict_int_to_char, 5: dict_int_to_char, 6: dict_int_to_char,
               2: dict_char_to_int, 3: dict_char_to_int}
    for j in [0, 1, 2, 3, 4, 5, 6]:
        if text[j] in mapping[j].keys():
            license_plate_ += mapping[j][text[j]]
        else:
            license_plate_ += text[j]

    return license_plate_

def read_license_plate_UK(license_plate_crop):
    detections = reader.readtext(license_plate_crop)

    for detection in detections:
        _, text, score = detection

        text = text.upper().replace(' ', '')

        if license_complies_format(text):
            return format_license(text), score
        
    return None, None

def read_license_plate(license_plate_crop):
    detections = reader.readtext(license_plate_crop)

    for detection in detections:
        _, text, score = detection

        text = text.upper().replace(' ', '')

        if text is not None:
            return text, score
    
    return None, None

def create_image(text, image_width=640, image_height=360, font_size = 128):
    image = Image.new('RGB', (image_width, image_height), color='white')
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=font_size)
    text_bbox = draw.textbbox((0, 0), text, font=font)

    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    text_x = (image_width - text_width) // 2
    text_y = (image_height - text_height) // 2

    draw.text((text_x, text_y), text, font=font, fill='black')

    return image

def draw_border(image, top_left, bottom_right, color=(0, 255, 0), thickness=10, line_length_x=200, line_length_y=200):
    
    x1, y1 = top_left
    x2, y2 = bottom_right

    cv2.line(image, (x1, y1), (x1, y1 + line_length_y), color, thickness)  
    cv2.line(image, (x1, y1), (x1 + line_length_x, y1), color, thickness)

    cv2.line(image, (x1, y2), (x1, y2 - line_length_y), color, thickness)  
    cv2.line(image, (x1, y2), (x1 + line_length_x, y2), color, thickness)

    cv2.line(image, (x2, y1), (x2 - line_length_x, y1), color, thickness)  
    cv2.line(image, (x2, y1), (x2, y1 + line_length_y), color, thickness)

    cv2.line(image, (x2, y2), (x2, y2 - line_length_y), color, thickness)  
    cv2.line(image, (x2, y2), (x2 - line_length_x, y2), color, thickness)

    return image