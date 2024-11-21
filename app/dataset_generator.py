import albumentations
import cv2
import csv
import random

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
def generate_sentence(word_list: list) -> str:
    selected_words = random.sample(word_list, 2)
    modified_words = [add_special_char(word) for word in selected_words]

    return f'{modified_words[0]} {modified_words[1]}'


# Функция преобразования рандомных ру-слов (словосочетаний) в png файлы (с наличием содержаний этих слов/словосочетаний)
def text_to_image(text: str, output_filename: str, font_size: int = 40, mode: str = 'b') -> None:
    based_path = f'pdf_appRecognizer/extract_assets/image_files/generated_assets/{output_filename}'

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
def test_dataset_generator():
    csv_path: str = 'pdf_appRecognizer/extract_assets/image_files/generated_assets/labels.csv'
    word_list: list = ru_words_lst()

    with open(file=csv_path, mode='a', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['filename', 'words'])

        for i in range(0, 1000):
            curr_text = generate_sentence(word_list=word_list)
            text_to_image(text=curr_text, output_filename=f'{i}.png')

            writer = csv.writer(csv_file)
            writer.writerow([f'{i}.png', curr_text])


# Вызов функции генератора датасета
test_dataset_generator()
