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
def ru_words_lst(src: str) -> list[str]:
    with open(file=src, mode='r', encoding='utf-8', errors='ignore') as file:
        word_list = [el.replace('\n', '') for el in file.readlines()]
        return word_list


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


def generate_preposition(word_list: list) -> str:
    selected_word: list = random.sample(word_list, 1)
    return selected_word[0]



def generate_inn_kpp_and_words(word_list: list) -> str:
    numbers_before_slash = ''.join(str(random.randint(0, 9)) for _ in range(10))
    numbers_after_slash = ''.join(str(random.randint(0, 9)) for _ in range(9))

    # Формируем итоговую строку c числами
    result = f"{numbers_before_slash}/{numbers_after_slash}"
    case = random.randint(1, 2)

    if case == 1:
        return result
    if case == 2:
        selected_words: list = random.sample(word_list, 2)
        return f'{selected_words[0]}/{selected_words[1]}'


def generate_words_symbols_with_special_chars(word_list: list) -> str:
    case: int = random.randint(1, 2)

    if case == 1:
        select_words: list = random.sample(word_list, 1)
        internal_case: int = random.randint(1, 2)

        if internal_case == 1:
            return f'{select_words[0]}:'

        if internal_case == 2:
            return f'{select_words[0]};'

    if case == 2:
        select_words: list = random.sample(word_list, 2)
        return f'{select_words[0]}, {select_words[1]}'


def numbers_with_special_chars_v1(word_list: list) -> str:
    case = random.randint(1, 6)

    if case == 1 or case == 2 or case == 3:
        # return f'{random_number_p1} {random_number_p2},{random_number_p3}'
        select_words: list = random.sample(word_list, 1)
        return f'({select_words[0]})'

    if case == 4:
        return f'({random.randint(1, 11)}а)'

    if case == 5:
        return f'({random.randint(1, 11)}в)'

    if case == 6:
        return f'({random.randint(1, 11)}б)'


def numbers_with_special_chars_v2() -> str:
    case = random.randint(1, 2)

    if case == 1:
        generated_random_nums: list = [str(random.randint(0, 9)) for i in range(0, random.randint(3, 7))]
        num_formatted: str = f'№ {''.join(generated_random_nums)}'
        return num_formatted

    if case == 2:
        generated_random_nums: list = [str(random.randint(1, 9)) for i in range(0, 2)]
        num_formatted: str = f'{''.join(generated_random_nums)}%'
        return num_formatted


def single_duo_chars() -> str:
    case = random.randint(1, 4)
    ru_alphabet: str = 'абвгдеёжзийклмнопрстуфхцчшщъыьэюя'

    if case == 1:
        return '№'

    if case  == 2:
        random_num: str = str(random.randint(10, 99))
        return random_num

    if case == 3:
        internal_case: int = random.randint(1, 3)

        if internal_case == 1:
            marks: str = ' « » '
            return marks

        if internal_case == 2:
            return  '«'

        if internal_case == 3:
            return '»'

    if case == 4:
        internal_case: int = random.randint(1, 2)

        random_index_1: int = random.randint(0, len(ru_alphabet)-1)
        random_index_2: int = random.randint(0, len(ru_alphabet) - 1)

        if internal_case == 1:
            return f'{ru_alphabet[random_index_1]}{ru_alphabet[random_index_2]}'

        if internal_case == 2:
            return f'{ru_alphabet[random_index_1]}.'


def generate_extended_data(word_list: list) -> str:
    ru_alphabet: str = 'абвгдеёжзийклмнопрстуфхцчшщъыьэюя'
    case: int = random.randint(1, 3)

    if case == 1:
        internal_case: int = random.randint(1, 3)

        if internal_case == 1:
            generated_ones_zeros_p1: list = [str(random.randint(0, 1)) for i in range(0, 2)]
            generated_ones_zeros_p2: list = [str(random.randint(0, 1)) for i in range(0, 2)]
            generated_ones_zeros_p3: list = [str(random.randint(0, 1)) for i in range(0, 2)]

            num_formatted: str = (f'№ 1{generated_ones_zeros_p1[0]}{generated_ones_zeros_p1[1]}1/1{generated_ones_zeros_p2[0]}{generated_ones_zeros_p2[1]}1/'
                                  f'1{generated_ones_zeros_p3[0]}{generated_ones_zeros_p3[1]}1')
            return num_formatted

        if internal_case == 2:
            generated_random_num_1: str = str(random.randint(1, 9))
            generated_random_num_2: str = str(random.randint(1, 9))

            return f'{generated_random_num_1}/{generated_random_num_2}'

        if internal_case == 3:
            random_index_1: int = random.randint(0, len(ru_alphabet) - 1)
            random_index_2: int = random.randint(0, len(ru_alphabet) - 1)

            return f'{ru_alphabet[random_index_1]}/{ru_alphabet[random_index_2]}'

    if case == 2:
        internal_case: int = random.randint(1, 3)

        if internal_case == 1:
            random_index_1: int = random.randint(0, len(ru_alphabet) - 1)
            random_index_2: int = random.randint(0, len(ru_alphabet) - 1)

            return f'{ru_alphabet[random_index_1].upper()}.{ru_alphabet[random_index_2].upper()}.'

        if internal_case == 2:
            extra_internal_case: int = random.randint(1, 2)
            random_index: int = random.randint(0, len(ru_alphabet) - 1)

            if extra_internal_case == 1:
                return f'{ru_alphabet[random_index]},'

            if extra_internal_case == 2:
                return f'{ru_alphabet[random_index]}.'

        if internal_case == 3:
            random_index_1: int = random.randint(0, len(ru_alphabet) - 1)
            random_index_2: int = random.randint(0, len(ru_alphabet) - 1)

            random_index_3: int = random.randint(0, len(ru_alphabet) - 1)
            random_index_4: int = random.randint(0, len(ru_alphabet) - 1)
            # Формат по типу "ИНН/КПП"
            return (f'{ru_alphabet[random_index_1].upper()}{ru_alphabet[random_index_2].upper()}/'
                    f'{ru_alphabet[random_index_3].upper()}{ru_alphabet[random_index_4].upper()}')

    if case == 3:
        internal_case: int = random.randint(1, 3)

        if internal_case == 1:
            selected_word: list = random.sample(word_list, 1)
            return f'({selected_word[0]})'

        if internal_case == 2:
            generated_random_num: str = str(random.randint(0, 9))
            return f'({generated_random_num})'

        if internal_case == 3:
            random_index_1: int = random.randint(0, len(ru_alphabet) - 1)
            random_index_2: int = random.randint(0, len(ru_alphabet) - 1)
            random_index_3: int = random.randint(0, len(ru_alphabet) - 1)

            return f'({ru_alphabet[random_index_1]}.{ru_alphabet[random_index_2]}.{ru_alphabet[random_index_3]}.)'


def generate_extended_data_v2(word_list: list) -> str:
    case: int = random.randint(3, 4)

    if case == 1:
        random_digit: str = str(random.randint(10, 20))
        return f'{random_digit}%'

    if case == 2:
        internal_case: int = random.randint(1, 2)
        selected_words: list = random.sample(word_list, 2)

        if internal_case == 1:
            return f'{selected_words[0]}-{selected_words[1]}'

        if internal_case == 2:
            return f'{selected_words[0].capitalize()}-{selected_words[1].capitalize()}'

    if case == 3:
        internal_case: int = random.randint(1, 6)
        selected_words: list = random.sample(word_list, 2)

        if internal_case == 1:
            return f'ООО «{selected_words[0]} {selected_words[1]}»'

        if internal_case == 2:
            return f'ООО «{selected_words[0].capitalize()} {selected_words[1].capitalize()}»'

        if internal_case == 3:
            return f'ООО "{selected_words[0].capitalize()} {selected_words[1].capitalize()}"'

        if internal_case == 4:
            return f'ООО "{selected_words[0]} {selected_words[1]}"'

        if internal_case == 5:
            return f'ООО \'{selected_words[0].capitalize()} {selected_words[1].capitalize()}\''

        if internal_case == 6:
            return f'ООО \'{selected_words[0]} {selected_words[1]}\''

    if case == 4:
        numbers_before_slash = ''.join(str(random.randint(0, 9)) for _ in range(10))
        numbers_after_slash = ''.join(str(random.randint(0, 9)) for _ in range(9))

        return f'{numbers_before_slash}/{numbers_after_slash}'



# Функция преобразования рандомных ру-слов (словосочетаний) в png файлы (с наличием содержаний этих слов/словосочетаний)
def text_to_image(text: str, output_filename: str, mode: str = 'b', is_dirty: bool = False) -> None:
    based_path: str = f'pdf_appRecognizer/extract_assets/image_files/generated_assets/extended_data/valid/images/{output_filename}'

    font_list: list[str] = ["arial.ttf", "arialbd.ttf", "ARIALNB.TTF", "ARIALN.TTF", "ariblk.ttf", "calibri.ttf",
                            "calibriz.ttf", "calibril.ttf", "cour.ttf", "courbd.ttf", "times.ttf", "timesbd.ttf",
                            "verdana.ttf", "verdanab.ttf"]

    font_size: int = random.randint(20, 40)
    max_height: int = 64
    max_width: int = int(font_size * len(text) * 1.1)

    image = Image.new('RGB', (max_width, max_height), 'white')
    draw = ImageDraw.Draw(image)

    try:
        random_font_index: int = random.randint(0, len(font_list)-1)
        font = ImageFont.truetype(font_list[random_font_index], font_size)
    except IOError:
        font = ImageFont.load_default()

    text_width, text_height = draw.textbbox((0, 0), text, font=font)[2:4]

    if text_height > max_height:
        text_height = max_height

    text_position = ((image.width - text_width) // 2, (max_height - text_height) // 2)
    draw.text(text_position, text, fill='black', font=font)

    img_array = np.array(image)

    if is_dirty:
        augmented_image = dirty_image(img_array, mode)
        blur_img = cv2.GaussianBlur(augmented_image, (5, 5), 0)

        dirty_img = Image.fromarray(blur_img)
        dirty_img.save(based_path)
    else:
        final_image = Image.fromarray(img_array)
        final_image.save(based_path)


# Генератор датасета
def run_dataset_generator(dataset_path: str, filename: str, dataset_count: int, data_img_name: str, src: str,
                          is_csv: bool = False, is_json: bool = False):
    word_list: list = ru_words_lst(src=src)

    if is_csv:
        with open(file=f'{dataset_path}/{filename}', mode='a', newline='') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(['filename', 'words'])

            for i in range(0, dataset_count):
                curr_text: str = numbers_with_special_chars_v2()
                cur_img_name: str = f'{i}_{data_img_name}.jpg'

                text_to_image(text=curr_text, output_filename=cur_img_name, is_dirty=True)

                writer = csv.writer(csv_file)
                writer.writerow([cur_img_name, curr_text])

    if is_json:
        label_dict: dict = {}

        for i in range(0, dataset_count):
            curr_text: str = generate_extended_data_v2(word_list=word_list)
            cur_img_name: str = f'{i}_{data_img_name}.jpg'

            text_to_image(text=curr_text, output_filename=cur_img_name)
            # label_dict[cur_img_name] = curr_text.replace(' ', '')
            label_dict[cur_img_name] = curr_text

        with open(file=f'{dataset_path}/{filename}', mode='w', encoding='utf-8') as file:
            json.dump(obj=label_dict, fp=file, ensure_ascii=False, indent=3)


# Вызов функции генератора датасета
run_dataset_generator(dataset_path='pdf_appRecognizer/extract_assets/image_files/generated_assets/extended_data/valid/images',
                      filename='extended_valid_v3_labels.json', dataset_count=6000, data_img_name='extended_data_v3', src='russian.txt',
                      is_json=True)

