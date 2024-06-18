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
                  'Стоимость товаров (работ, услуг), имущественных прав с налогом всего')


class DataCleaning:
    """
    Статический класс, предназначенный для работы с таблицей данных (очистка/удаление лишних строк (списков)/столбцов)
    """
    @staticmethod
    def data_clean(data_table: list) -> list:
        logger.info('Данные до очистки: \n%s', data_table)
        # Удаляем последнюю строку (Всегда информационный мусор), а также первую (которая не входит в табличную часть)
        cleaned_table = data_table[1:len(data_table)-1]
        logger.info('Очистка - стадия 1: \n%s', cleaned_table)
        # Очищаем таблицу от лишних отступов и переходов
        cleaned_table_2 = [[item.replace('\n', ' ').replace(' /', '/').replace('- ', '').replace('/ ', '/').
                            replace(',', ', ').replace('  ', ' ') for item in sublist] for sublist in cleaned_table]
        logger.info('Очистка - стадия 2: \n%s', cleaned_table_2)
        # создаем компаратор на основе кортежа NEEDLE_COLUMNS для того, чтобы сравнивать преобразованные элементы
        comparator_columns: list = list(NEEDLE_COLUMNS)
        comparator_columns = [el.replace(' ', '').lower() for el in comparator_columns]
        logger.info("создали компаратор для сравнения названия столбцов \n%s", comparator_columns)
        # Создаем список индексов, туда будем добавлять индексы нужных колонок (которые совпали с компаратором)
        indexes: list[int] = []
        # Начинаем сравнивать
        for lst in cleaned_table_2:
            for i in range(len(lst)):
                el = lst[i].replace('-', '').replace('–', '').replace(' ', '').lower()
                # Строку сверху я буду потом дорабатывать
                if el in comparator_columns:
                    indexes.append(i)

        logger.info("Номера индексов: %s\nКол-во индексов (должно быть 6): %s", indexes, len(indexes))
        # На основе полученных индексов, убираем ненужные столбцы
        result = [[sublist[i] for i in indexes] for sublist in cleaned_table_2]
        logger.info('Очистка - стадия 3: \n%s', result)
        # Очищаем от пустых значений
        final_result = [el for el in result if sum(map(len, el)) > 0]
        logger.info('Финальная очитка (почти): \n%s', final_result)
        return final_result


class DataCollection:
    """
    Класс для преобразования (записи) извлеченных данных из pdf в коллекцию (хэш-таблицу)
    """
    def __init__(self):
        self.data: dict = {}

    def data_collect(self, inn_kpp: str, invoice: str, data_table: str, total: str) -> dict | Exception:
        try:
            self.data['inn_kpp'] = inn_kpp
            logger.info('Записали ИНН/КПП: %s', self.data)
            self.data['invoice'] = invoice
            logger.info('Записали счет-фактуру: %s', self.data)
            self.data['data_table'] = data_table
            logger.info('Записали таблицу данных: %s', self.data)
            self.data['total'] = total
            logger.info('JSON\'s data (все записано, но до очистки): %s', self.data)
            return self.data
        except IndexError as e:
            logger.error('Ошибка - %s', e)
            return e


class DictToJson:
    """
    Класс для записи хэш-таблицы в json файл
    """
    @staticmethod
    def write_to_json(collection: dict) -> None:
        with open(file='extracted_results/data.json', mode='w', encoding='utf-8') as file:
            json.dump(obj=collection, fp=file, ensure_ascii=False, indent=3)