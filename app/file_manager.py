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


def move_jpg_files(source_folder, destination_folder):

    # Проверка существования папок
    if not os.path.exists(source_folder):
        print("Исходная папка не существует")
        return
    if not os.path.exists(destination_folder):
        print("Папка назначения не существует")
        return

    # Перемещение файлов
    for filename in os.listdir(source_folder):
        if filename.endswith('.jpg'):
            source_file = os.path.join(source_folder, filename)
            destination_file = os.path.join(destination_folder, filename)

            shutil.move(source_file, destination_file)
            # print(f"Файл {filename} перемещен в {destination_folder}")


def get_name_images(path_to_json: str):

    with open(file=path_to_json, mode='r', encoding='utf-8') as cur_json_file:
        cur_data: dict = json.load(cur_json_file)
        cur_key_list = cur_data.keys()

    return cur_key_list


