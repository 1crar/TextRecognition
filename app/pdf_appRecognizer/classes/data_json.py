import json
import re
from logging import getLogger

logger = getLogger(__name__)
# Необходимые наименования колонок для извлечения
NEEDLE_COLUMNS = ('Наименование товара (описание выполненных работ, оказанных услуг), имущественного права',
                  'Количество (объем)',
                  'Цена (тариф) за единицу измерения',
                  'Стоимость товаров (работ, услуг), имущественных прав без налога всего',
                  'Налоговая ставка',
                  'Сумма налога, предъявляемая покупателю',
                  'Стоимость товаров (работ, услуг), имущественных прав с налогом всего')


class DataCleaning:
    """
    Статический класс, предназначенный для работы с таблицей данных (очистка/удаление лишних строк (списков)/столбцов)
    """
    @staticmethod
    def data_clean(data_table: list) -> list:
        logger.info('Данные до очистки: \n%s', data_table)
        # Очищаем таблицу от лишних отступов и переходов
        cleaned_table = [[item.replace('\n', ' ').replace('--', '').replace(' /', '/').replace('- ', '').replace('/ ', '/').
                          replace(',', ', ').replace('  ', ' ') for item in sublist] for sublist in data_table]
        logger.info('Очистка - стадия 1: \n%s', cleaned_table)
        # Убираем начальное n-ое количество строк, которое не является частью таблицы
        cleaned_table_2 = first_row_check(data_table=cleaned_table)
        logger.info('Очистка - стадия 2: \n%s', cleaned_table)
        # Убираем конечное n-ое количество строк, которое не является частью таблицы
        cleaned_table_3 = last_row_check(data_table=cleaned_table_2)
        logger.info('Очистка - стадия 3: \n%s', cleaned_table_3)
        # создаем компаратор на основе кортежа NEEDLE_COLUMNS для того, чтобы сравнивать преобразованные элементы
        comparator_columns: list = list(NEEDLE_COLUMNS)
        comparator_columns = [el.replace(' ', '').lower() for el in comparator_columns]
        logger.info("создали компаратор для сравнения названия столбцов \n%s", comparator_columns)
        # Создаем список индексов, туда будем добавлять индексы нужных колонок (которые совпали с компаратором)
        indexes: list[int] = []
        # Создаем список названия колонок для сравнения с компаратором
        column_names: list = cleaned_table_3[0]
        # Начинаем сравнивать
        for i, name in enumerate(column_names):
            el = name.replace('-', '').replace('–', '').replace(' ', '').lower()
            if el in comparator_columns:
                indexes.append(i)

        logger.info("Номера индексов: %s\nКол-во индексов (должно быть 7): %s", indexes, len(indexes))
        # На основе полученных индексов, убираем ненужные столбцы
        result = [[sublist[i] for i in indexes] for sublist in cleaned_table_3]
        logger.info('Очистка - стадия 4: \n%s', result)
        # Очищаем от пустых значений
        pre_final_result = [el for el in result if sum(map(len, el)) > 0]
        # Очищаем от дубликатов (дублирующих строк, такие могут быть)
        final_result: list = []
        for lst in pre_final_result:
            if lst not in final_result:
                final_result.append(lst)

        # Очищаем от возможной следующей строки (списка): ['1а', '3', '4', '5', '7', '9']
        garbage_row = ['1а', '3', '4', '5', '7', '9']
        # Так как 1а определяется в каждой табличной части УПД файлов, то делаем следующее
        garbage_detecter: str = garbage_row[0]
        for i, lst in enumerate(final_result):
            if garbage_detecter in lst:
                final_result.pop(i)
        # Выводим очищенный список (табличную часть)
        logger.info('Финальная очитка: \n%s', final_result)
        return final_result


class DataCollection:
    """
    Класс для преобразования (записи) извлеченных и преобразованных данных из pdf в коллекцию (хэш-таблицу)
    """
    def __init__(self):
        self.data: dict = {}

    def data_collect(self, inn_kpp: str, invoice: str, cleaned_data: list, totals: tuple) -> dict | Exception:
        try:
            self.data['inn_kpp'] = inn_kpp
            logger.info('Записали ИНН/КПП: %s', self.data)
            self.data['invoice'] = invoice
            logger.info('Записали счет-фактуру: %s', self.data)
            # Убираем последнюю строку, так как эта строка с суммой (сумма с налогом/без налога, сумма налога)
            self.data['data_table'] = cleaned_data[:len(cleaned_data)-1]
            logger.info('Записали таблицу данных: %s', self.data)
            self.data['total_without_tax'] = totals[0]
            self.data['amount_of_tax'] = totals[1]
            self.data['total_amount'] = totals[2]
            logger.info('JSON\'s data (все записано): %s', self.data)
            return self.data
        except IndexError as e:
            logger.error('Ошибка - %s', e)
            return e


class DictToJson:
    """
    Класс для записи хэш-таблицы в json файл
    """
    @staticmethod
    def write_to_json(collection: dict, path_to_save: str) -> None:
        with open(file=path_to_save, mode='w', encoding='utf-8') as file:
            json.dump(obj=collection, fp=file, ensure_ascii=False, indent=3)


def first_row_check(data_table: list) -> list:
    counter_to_del: int = 0
    for row in data_table:
        if any(el in NEEDLE_COLUMNS for el in row):
            break
        counter_to_del += 1

    new_table = data_table[counter_to_del:]
    return new_table


def last_row_check(data_table: list) -> list:
    id_word = 'всегокоплате'
    counter_to_del: int = 0

    for row in reversed(data_table):
        # Строка, проверяющая наличия id_word в списке (строке) data_table
        if any(el.replace(' ', '').replace(':', '').lower()[:len(id_word)] == id_word for el in row):
            break
        counter_to_del += 1

    new_table = data_table[:len(data_table)-counter_to_del]
    return new_table


