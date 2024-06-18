import json
from logging import getLogger

logger = getLogger(__name__)
NEEDLE_COLUMNS = ('Наименование товара (описание выполненных работ, оказанных услуг), имущественного права',
                  'Количество (объем)',
                  'Цена (тариф) за единицу измерения',
                  'Стоимость товаров (работ, услуг), имущественных прав без налога всего',
                  'Налоговая ставка',
                  'Стоимость товаров (работ, услуг), имущественных прав с налогом всего')


class DataCleaning:
    @staticmethod
    def data_clean(data_table: list) -> list:
        logger.info('Данные до очистки: \n%s', data_table)
        # Удаляем последнюю строку (Всегда информационный мусор), а также первую (которая не входит в табличную часть)
        cleaned_table = data_table[1:len(data_table)-1]
        logger.info('Очистка - стадия 1: \n%s', cleaned_table)
        # Очищаем таблицу от лишних отступов и переходов
        cleaned_table_2 = [[item.replace('\n', ' ').replace('/ ', '/').replace('- ', '').replace(' /', '/').
                            replace(',', ', ').replace('  ', ' ') for item in sublist] for sublist in cleaned_table]
        logger.info('Очистка - стадия 2: \n%s', cleaned_table_2)
        return cleaned_table


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