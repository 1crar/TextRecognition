import json
from logging import getLogger

logger = getLogger(__name__)


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