import string
import easyocr

# OCR 리더 초기화
reader = easyocr.Reader(['ko'], gpu=False)

# 한국 번호판의 유효한 문자 목록
char_list = [
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    '가', '강', '거', '경', '고', '광', '구', '기',
    '나', '남', '너', '노', '누',
    '다', '대', '더', '도', '동', '두',
    '라', '러', '로', '루',
    '마', '머', '모', '무', '문',
    '바', '배', '버', '보', '부', '북',
    '사', '산', '서', '세', '소', '수',
    '아', '어', '오', '우', '울', '원', '육', '인',
    '자', '저', '전', '제', '조', '종', '주',
    '천', '충',
    '하', '허', '호'
]

def write_csv(results, output_path):
    """
    Write the results to a CSV file.

    Args:
        results (dict): Dictionary containing the results.
        output_path (str): Path to the output CSV file.
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('{},{},{},{},{},{},{}\n'.format('frame_nmr', 'car_id', 'car_bbox',
                                                'license_plate_bbox', 'license_plate_bbox_score', 'license_number',
                                                'license_number_score'))

        for frame_nmr in results.keys():
            for car_id in results[frame_nmr].keys():
                print(results[frame_nmr][car_id])
                if 'car' in results[frame_nmr][car_id].keys() and \
                   'license_plate' in results[frame_nmr][car_id].keys() and \
                   'text' in results[frame_nmr][car_id]['license_plate'].keys():
                    f.write('{},{},{},{},{},{},{}\n'.format(frame_nmr,
                                                            car_id,
                                                            '[{} {} {} {}]'.format(
                                                                results[frame_nmr][car_id]['car']['bbox'][0],
                                                                results[frame_nmr][car_id]['car']['bbox'][1],
                                                                results[frame_nmr][car_id]['car']['bbox'][2],
                                                                results[frame_nmr][car_id]['car']['bbox'][3]),
                                                            '[{} {} {} {}]'.format(
                                                                results[frame_nmr][car_id]['license_plate']['bbox'][0],
                                                                results[frame_nmr][car_id]['license_plate']['bbox'][1],
                                                                results[frame_nmr][car_id]['license_plate']['bbox'][2],
                                                                results[frame_nmr][car_id]['license_plate']['bbox'][3]),
                                                            results[frame_nmr][car_id]['license_plate']['bbox_score'],
                                                            results[frame_nmr][car_id]['license_plate']['text'],
                                                            results[frame_nmr][car_id]['license_plate']['text_score'])
                            )
        f.close()


def license_complies_format(text):
    """
    번호판 텍스트가 한국 번호판 형식을 따르는지 확인합니다.

    Args:
        text (str): 번호판 텍스트.

    Returns:
        bool: 번호판 형식을 따를 경우 True, 아니면 False.
    """
    # 한글 문자인지 확인하는 조건 (한글 유니코드 범위)
    def is_korean_character(char):
        return '가' <= char <= '힣'

    # 한국 번호판은 길이에 따라 유효한 형식이 다양함
    if len(text) == 7:  # 예: "12가3456"
        if text[:2].isdigit() and is_korean_character(text[2]) and text[3:].isdigit():
            return True
    elif len(text) == 8:  # 예: "123가4567"
        if text[:3].isdigit() and is_korean_character(text[3]) and text[4:].isdigit():
            return True
    return False


def format_license(text):
    """
    번호판 텍스트를 정리하고 한국 번호판 형식에 맞게 반환합니다.

    Args:
        text (str): 번호판 텍스트.

    Returns:
        str: 정리된 번호판 텍스트.
    """
    # 한글 문자인지 확인하는 조건 (한글 유니코드 범위)
    def is_korean_character(char):
        return '가' <= char <= '힣'

    # 인식된 텍스트에서 유효하지 않은 문자를 제거
    formatted_text = ''.join([char for char in text if char.isdigit() or is_korean_character(char)])
    return formatted_text



def read_license_plate(license_plate_crop):
    """
    번호판 이미지에서 텍스트를 인식합니다.

    Args:
        license_plate_crop (PIL.Image.Image): 번호판이 포함된 이미지.

    Returns:
        tuple: 번호판 텍스트와 신뢰도 점수.
    """
    detections = reader.readtext(license_plate_crop)

    for detection in detections:
        bbox, text, score = detection
        text = text.replace(' ', '')  # 공백 제거
        if license_complies_format(text):
            return format_license(text), score

    return None, None



def get_car(license_plate, vehicle_track_ids):
    """
    번호판 좌표를 기준으로 차량 좌표와 ID를 반환합니다.

    Args:
        license_plate (tuple): 번호판 좌표 (x1, y1, x2, y2, score, class_id).
        vehicle_track_ids (list): 차량 좌표와 ID 리스트.

    Returns:
        tuple: 차량 좌표 (x1, y1, x2, y2)와 ID.
    """
    x1, y1, x2, y2, score, class_id = license_plate

    for j, (xcar1, ycar1, xcar2, ycar2, car_id) in enumerate(vehicle_track_ids):
        if x1 > xcar1 and y1 > ycar1 and x2 < xcar2 and y2 < ycar2:
            return xcar1, ycar1, xcar2, ycar2, car_id

    return -1, -1, -1, -1, -1
