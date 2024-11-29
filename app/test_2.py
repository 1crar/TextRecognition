import cv2

from dotenv import load_dotenv
from img2table.document import Image
from PIL import Image, ImageDraw, ImageFont

import torch
import easyocr
import numpy as np

from img2table.ocr import EasyOCR
from PIL import Image as Img

load_dotenv()


enhanced_file: str = 'temp/scaled4_loveimg_test_2.png'
image_path: str = f'pdf_appRecognizer/extract_assets/image_files/dt_cells/table_cell_0_3266_555.png'

image_cv2 = cv2.imread(filename=image_path)
languages: list = ['ru']

# dt_img2excel(img_path=image_path)


print("[INFO] OCR'ing input image...")
print(f"[INFO] OCR'ing with the following languages: {languages}")

test_case: int = 2

if test_case == 1:
    reader = easyocr.Reader(lang_list=languages, gpu=True,
                            model_storage_directory='fine_tune', user_network_directory='fine_tune',
                            recog_network='ru_finetune')
    # results = reader.readtext(image_cv2, text_threshold=0.75, contrast_ths=0.7, width_ths=1.25, height_ths=0.75,
    #                           ycenter_ths=0.5, slope_ths=1, add_margin=0.175, decoder='wordbeamsearch', beamWidth=20,
    #                           canvas_size=3500)
    # Not bad
    results = reader.readtext(image_cv2, text_threshold=0.75, contrast_ths=0.8, width_ths=1.3, height_ths=0.75,
                              ycenter_ths=0.5, slope_ths=1, add_margin=0.200, decoder='wordbeamsearch', beamWidth=20,
                              canvas_size=3500)
if test_case == 2:
    reader = easyocr.Reader(lang_list=languages, gpu=True)
    # results = reader.readtext(image_cv2)
    results = reader.readtext(image_cv2)

# Преобразуем изображение OpenCV в формат PIL
image_pil = Image.fromarray(cv2.cvtColor(image_cv2, cv2.COLOR_BGR2RGB))
draw = ImageDraw.Draw(image_pil)

# Выберите шрифт (если нужный шрифт не установлен, вы можете указать путь к .ttf файлу)
font = ImageFont.truetype(font="arial.ttf", size=40)  # Убедитесь, что путь к шрифту корректный

for (bbox, text, prob) in results:
    print("[INFO] {:.4f}: {}".format(prob, text))
    # Распаковка координат
    (tl, tr, br, bl) = bbox
    tl = (int(tl[0]), int(tl[1]))
    br = (int(br[0]), int(br[1]))

    # Рисуем прямоугольник
    draw.rectangle([tl, br], outline=(0, 255, 0), width=2)

    # Рисуем текст
    draw.text((tl[0], tl[1] - 10), text, fill=(0, 255, 0), font=font)

# Отображаем результат
image_pil.show()

# Сохранение изображения (необязательно)
# image_pil.save('test_3.png')


# reader = easyocr.Reader(['ru'])
#
# img = Image.open(fp='pdf_appRecognizer/extract_assets/image_files/YPD_1/enhance_ver2_UPD_1_scaled_3.png')
#
# gray_img = img.convert('L')
# gray_img = cv2.imread('pdf_appRecognizer/extract_assets/image_files/YPD_1/enhance_ver2_UPD_1_scaled_3.png')
#
# result = reader.readtext(gray_img, detail=0)

# extracted_data: dict = {
#     'extracted_data': result
# }
#
# with open(file='pdf_appRecognizer/extract_assets/json_files/extracted_data_easyOCR.json',
#           mode='w', encoding='utf-8') as json_file:
#     json.dump(obj=extracted_data, fp=json_file, ensure_ascii=False, indent=3)
#
# print(result)


# text_to_image(text="Привет, мир!", font_size=40, output_filename='1.png')
# dirty_image(img='pdf_appRecognizer/extract_assets/image_files/generated_assets/0.png', mode='b')
