import json
import os
import shutil



def merge_json_files(folder_path, output_file):
    merged_data = {}

    for filename in os.listdir(folder_path):
        if filename.endswith('.json'):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # Добавление данных в словарь. Используйте имя файла как ключ, если это нужно.
                merged_data[filename] = data

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(merged_data, f, ensure_ascii=False, indent=3)


# Пример использования:
# merge_json_files(r'C:\Users\serge\Рабочий стол\all_labels', 'labels.json')



def move_jpg_file(source_directory, target_directory, file_name_list):

    for file_name in file_name_list:
        # Полный путь к исходному файлу
        source_file = os.path.join(source_directory, file_name)
        # Полный путь к новому местоположению
        target_file = os.path.join(target_directory, file_name)

        # Перемещаем скопированный .jpg-файл
        shutil.copy2(source_file, target_file)
        # print(f"Файл {file_name} успешно перемещен в {target_directory}.")


def get_name_images(path_to_json: str):
    with open(file=path_to_json, mode='r', encoding='utf-8') as cur_json_file:
        cur_data: dict = json.load(cur_json_file)
        cur_key_list = cur_data.keys()

    return cur_key_list


path_to_validate_json: str = 'pdf_appRecognizer/extract_assets/image_files/generated_assets/validate_dataset/labels.json'
jpg_file_names = get_name_images(path_to_json=path_to_validate_json)

move_jpg_file(source_directory='pdf_appRecognizer/extract_assets/image_files/generated_assets/union_dataset/images',
              target_directory='pdf_appRecognizer/extract_assets/image_files/generated_assets/validate_dataset/images',
              file_name_list=jpg_file_names)



# path_dataset: str = 'pdf_appRecognizer/extract_assets/image_files/generated_assets/union_dataset/images'
# for img_file in os.listdir(path_dataset):
#     file_path = os.path.join(path_dataset, img_file)
#
#     img_cv2 = cv2.imread(filename=file_path)
#     if img_cv2.shape[0] == 0 or img_cv2.shape[1] == 0:
#         print(file_path)
#
# json_file_path: str = r'C:\Users\serge\Рабочий стол\all_labels\6.json'
# new_json_file_path: str = r'C:\Users\serge\Рабочий стол\all_labels\6_1.json'
#
# dataset_path: str = 'pdf_appRecognizer/extract_assets/image_files/generated_assets/union_dataset/images'
# cropped_label: dict = {}
#
# with open(file=json_file_path, mode='r', encoding='utf-8') as json_file:
#     data: dict = json.load(json_file)
#     key_list = data.keys()
#
#     for cur_img_file in os.listdir(dataset_path):
#         if cur_img_file in key_list:
#             cropped_label[cur_img_file] = data[cur_img_file]
#
#
# with open(file=new_json_file_path, mode='w', encoding='utf-8') as new_json_file:
#     json.dump(obj=cropped_label, fp=new_json_file, ensure_ascii=False, indent=3)