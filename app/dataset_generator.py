import albumentations
import cv2
import csv
import random
import json

import numpy as np

from PIL import Image, ImageDraw, ImageFont


# Функция генерация шумов для изображений
def dirty_image(img, mode):
    aug = None

    if mode == 'b':
        aug = albumentations.CoarseDropout(max_holes=120, max_height=3, max_width=3, min_holes=30,
                                           min_width=1, min_height=1,
                                           fill_value=15, p=1)
    elif mode == 'w':
        aug = albumentations.CoarseDropout(max_holes=1500, max_height=2, max_width=2, min_holes=300,
                                           min_height=1, min_width=1,
                                           fill_value=225, p=1)

    transform = albumentations.Compose([aug])
    img = transform(image=img)['image']
    return img


# Функция считывания ру слов из текстового файла 10000-russian-words.txt
def ru_words_lst() -> list[str]:
    with open(file='10000-russian-words.txt', mode='r', encoding='utf-8') as file:
        word_list = [el.replace('\n', '') for el in file.readlines()]
        return word_list


# Функция, которая рандомно добавляет спец-символы в ру-слова (нужно для дообучения)
def add_special_char(word) -> str:
    specials = '/№()\"\"«»'

    if random.choice([True, False]):
        special_char = random.choice(specials)
        position = random.randint(0, len(word))
        return word[:position] + special_char + word[position:]

    return word


# Рандомный генератор слов со спецсимволами
def generate_name_company(word_list: list) -> str:
    selected_words: list = random.sample(word_list, 2)
    # modified_words = [add_special_char(word) for word in selected_words]
    case = random.randint(1, 3)

    # no switches :(
    if case == 1:
        return f'ООО «{selected_words[0].capitalize()} {selected_words[1].capitalize()}»'
    if case == 2:
        return f'ООО "{selected_words[0].capitalize()} {selected_words[1].capitalize()}"'
    if case == 3:
        return f'ООО \'{selected_words[0].capitalize()} {selected_words[1].capitalize()}\''


def generate_inn_kpp_and_numbers() -> str:
    numbers_before_slash = ''.join(str(random.randint(0, 9)) for _ in range(10))
    numbers_after_slash = ''.join(str(random.randint(0, 9)) for _ in range(9))

    # Формируем итоговую строку
    result = f"{numbers_before_slash}/{numbers_after_slash}"
    case = random.randint(1, 4)

    if case == 1 or case == 2:
        return result
    if case == 3:
        return 'ИНН/КПП продавца:'
    if case == 4:
        return 'ИНН/КПП покупателя:'


def generate_words_with_special_chars() -> str:
    random_day = random.randint(10, 30)
    random_month = random.randint(10, 12)

    special_sentence: str = (f'№ п.п {random.randint(0, 9)}-{random.randint(0, 9)} № '
                             f'{random.randint(1, 9999)} от {random_day}.{random_month}.2025')
    case = random.randint(1, 4)

    if case == 1 or case == 2:
        return special_sentence
    if case == 3:
        return 'Счет-фактура №'
    if case == 4:
        return 'Исправления №'


def numbers_with_special_chars() -> str:
    random_number_p1: str = str(random.randint(1, 9))
    random_number_p2: str = str(random.randint(1, 999))
    random_number_p3: str = str(random.randint(10, 999))

    case = random.randint(1, 4)
    if case == 1 or case == 2:
        return f'{random_number_p1} {random_number_p2},{random_number_p3}'
    if case == 3:
        return f'({random.randint(1, 11)}а)'
    if case == 4:
        return f'({random.randint(1, 11)}в)'


# Функция преобразования рандомных ру-слов (словосочетаний) в png файлы (с наличием содержаний этих слов/словосочетаний)
def text_to_image(text: str, output_filename: str, font_size: int = 40, mode: str = 'b') -> None:
    based_path = f'pdf_appRecognizer/extract_assets/image_files/generated_assets/numbers_with_special_chars/{output_filename}'

    max_height = 64
    max_width = int(font_size * len(text) * 0.6)

    image = Image.new('RGB', (max_width, max_height), 'white')
    draw = ImageDraw.Draw(image)

    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except IOError:
        font = ImageFont.load_default()

    text_width, text_height = draw.textbbox((0, 0), text, font=font)[2:4]

    if text_height > max_height:
        text_height = max_height

    text_position = ((image.width - text_width) // 2, (max_height - text_height) // 2)
    draw.text(text_position, text, fill='black', font=font)

    img_array = np.array(image)
    augmented_image = dirty_image(img_array, mode)

    blur_img = cv2.GaussianBlur(augmented_image, (5, 5), 0)
    final_image = Image.fromarray(blur_img)

    final_image.save(based_path)


# Генератор датасета
def run_dataset_generator(dataset_path: str, filename: str, dataset_count: int, is_csv: bool = False,
                          is_json: bool = False,):
    # word_list: list = ru_words_lst()

    if is_csv:
        with open(file=f'{dataset_path}/{filename}', mode='a', newline='') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(['filename', 'words'])

            for i in range(0, dataset_count):
                curr_text = numbers_with_special_chars()
                text_to_image(text=curr_text, output_filename=f'{i}.png')

                writer = csv.writer(csv_file)
                writer.writerow([f'{i}_nums_special_chars.png', curr_text])

    if is_json:
        label_dict: dict = {}

        for i in range(0, dataset_count):
            curr_text: str = numbers_with_special_chars()
            cur_img_name: str = f'{i}_nums_special_chars.png'

            text_to_image(text=curr_text, output_filename=cur_img_name)
            label_dict[cur_img_name] = curr_text.replace(' ', '')

        with open(file=f'{dataset_path}/{filename}', mode='w', encoding='utf-8') as file:
            json.dump(obj=label_dict, fp=file, ensure_ascii=False, indent=3)


# Вызов функции генератора датасета
run_dataset_generator(dataset_path='pdf_appRecognizer/extract_assets/image_files/generated_assets/numbers_with_special_chars',
                      filename='label.json', dataset_count=18000, is_json=True)

