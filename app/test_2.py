import os
import shutil
import json
import cv2

from dotenv import load_dotenv
from img2table.document import Image
from PIL import Image, ImageDraw, ImageFont

from pdf_appRecognizer.classes.img import generate_csv_file

import easyocr

load_dotenv()


def remove_column_names(extracted_data: list) -> list:
    cleaned_data: list = []
    is_flag = False

    for cur_text in extracted_data:
        if cur_text == 'A' or cur_text == 'А':
            is_flag = True

        if is_flag:
            cleaned_data.append(cur_text)
    return cleaned_data


def is_intersecting(bbox1, bbox2):
    (tl1, tr1, br1, bl1) = bbox1
    (tl2, tr2, br2, bl2) = bbox2

    if tl1[0] < br2[0] and br1[0] > tl2[0] and tl1[1] < br2[1] and br1[1] > tl2[1]:
        return True
    return False


def find_intersecting_bboxes(index, results, found_indices):
    bbox1, text1, prob1 = results[index]
    for i in range(len(results)):
        if i not in found_indices and is_intersecting(bbox1, results[i][0]):
            found_indices.add(i)
            find_intersecting_bboxes(i, results, found_indices)


def main_test_2(folder_path: str, img_filename: str, test_case: int, is_complex_row: bool):
    image_cv2 = cv2.imread(filename=f'{folder_path}/{img_filename}')
    languages: list = ['ru']

    print("[INFO] OCR'ing input image...")
    print(f"[INFO] OCR'ing with the following languages: {languages}")

    results: list = []

    if test_case == 1:
        reader = easyocr.Reader(lang_list=languages, gpu=True)
        results = reader.readtext(image_cv2, text_threshold=0.0, contrast_ths=0.8, width_ths=1.2, height_ths=1.9,
                                  ycenter_ths=0.5, slope_ths=1, add_margin=0.1, decoder='wordbeamsearch', beamWidth=20,
                                  canvas_size=3500, y_ths=0.8, blocklist='[]|/_')
    if test_case == 2:
        reader = easyocr.Reader(lang_list=languages, gpu=True)
        # results = reader.readtext(image_cv2)
        results = reader.readtext(image_cv2)

    image_pil = Image.fromarray(cv2.cvtColor(image_cv2, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image_pil)

    font = ImageFont.truetype(font="arial.ttf", size=20)

    data_table: list = []

    filtered_results = []
    used_indices = set()

    for i in range(len(results)):
        if i not in used_indices:
            current_indices = set([i])

            find_intersecting_bboxes(i, results, current_indices)
            used_indices.update(current_indices)
            # Добавляем все найденные bbox в filtered_results
            for idx in current_indices:
                filtered_results.append(results[idx])

    if is_complex_row:
        for i in range(len(results)):
            if i not in used_indices:
                current_indices = set([i])

                find_intersecting_bboxes(i, results, current_indices)
                used_indices.update(current_indices)
                # Добавляем все найденные bbox в filtered_results
                for idx in current_indices:
                    filtered_results.append(results[idx])

        for (bbox, text, prob) in filtered_results:
            print("[INFO] {:.4f}: {}".format(prob, text))
            # Распаковка координат
            (tl, tr, br, bl) = bbox
            tl = (int(tl[0]), int(tl[1]))
            br = (int(br[0]), int(br[1]))

            # Рисуем прямоугольник
            draw.rectangle([tl, br], outline=(0, 255, 0), width=2)

            # Рисуем текст
            draw.text((tl[0], tl[1] - 10), text, fill=(0, 255, 0), font=font)
            data_table.append(text)
    else:
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
            data_table.append(text)

    cleaned_data_table: list = remove_column_names(extracted_data=data_table)
    generate_csv_file(table=cleaned_data_table,
                      csv_filename=f'pdf_appRecognizer/extract_assets/image_files/YPDs/test_2/csv_files/'
                                   f'{img_filename[:-4]}.csv')

    # Отображаем результат
    image_pil.show()


def main_test_3_cell_extracter():
    cropped_path: str = ('pdf_appRecognizer/extract_assets/image_files/YPDs/test_1/dt_cropped/cropped_cells/'
                         'cells_99_cropped')
    detected_files_path = os.path.join(cropped_path, 'detected_files')
    # Создание папки 'detected_files', если она не существует
    os.makedirs(detected_files_path, exist_ok=True)

    previous_file_name = None  # Переменная для хранения имени предыдущего файла
    list_text: list = []     # Список для хранения текста

    counter_file: int = 0
    labels: dict = {}       # Словарь для записи лейблов в JSON

    for cropped_cell_file in os.listdir(path=cropped_path):

        cur_full_path = os.path.join(cropped_path, cropped_cell_file)

        if os.path.isfile(cur_full_path):
            cell_cv2 = cv2.imread(filename=cur_full_path)
            languages: list = ['ru']

            reader = easyocr.Reader(lang_list=languages, gpu=True)
            results = reader.readtext(cell_cv2)

            for (_, text, _) in results:
                # print(f'INFO:  извлеченный текст:\t{text}   из файла:\t{cropped_cell_file}')
                # counter_file += 1

                # Сравнение с именем предыдущего файла
                if previous_file_name is not None:
                    if cropped_cell_file == previous_file_name:
                        list_text.append(text)
                    else:
                        list_text.append(text)
                        union_text = ' '.join(list_text)

                        print(f'INFO: Извлеченный текст:\t{union_text}  из файла:\t{cropped_cell_file}')
                        # print(f'INFO: Имена файлов различаются: {cropped_cell_file} и {previous_file_name}.')
                        list_text = []
                        labels[cropped_cell_file] = union_text

                        shutil.copy(cur_full_path, detected_files_path)
                        counter_file += 1
                else:
                    print(f'INFO: Извлеченный текст:\t{text}  из файла:\t{cropped_cell_file}')
                    labels[cropped_cell_file] = text

                    shutil.copy(cur_full_path, detected_files_path)
                    counter_file += 1

                previous_file_name = cropped_cell_file  # Обновление имени предыдущего файла

    with open(file=f'{detected_files_path}/labels.json', mode='w', encoding='utf-8') as file:
        json.dump(obj=labels, fp=file, ensure_ascii=False, indent=3)

    print(f'INFO: Кол-во файлов\t{counter_file}')


# main_test_3_cell_extracter()

# cur_folder_path: str = 'pdf_appRecognizer/extract_assets/image_files/YPDs/test_2/dt_cropped'
# cur_img_file: str = 'YPD_2_cropped.png'
# # Запуск функции извлечения данных из обрезанной таблицы из УПД (с границами/без границ ячеек)
# main_test_2(folder_path=cur_folder_path, img_filename=cur_img_file, test_case=1, is_complex_row=True)

