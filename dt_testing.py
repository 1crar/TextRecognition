import json
from classes.text_extracter import DictToJson

with open(file='extracted_results/data.json', mode='r', encoding='utf-8') as file:
    data = json.load(file)


data_table: list = data['data_table'][1:]
print(f'До очистки --- {data_table}')

cleaned_data = [[item.replace('\n', ' ').replace('/ ', '/').replace('- ', '').replace(' /', '/').
                 replace(',', ', ').replace('  ', ' ') for item in sublist] for sublist in data_table]

total_list: list = cleaned_data[len(cleaned_data)-1]

print(f'После очистки --- {cleaned_data}')

needle_columns = ('Наименование товара (описание выполненных работ, оказанных услуг), имущественного права',
                  'Количество (объем)',
                  'Цена (тариф) за единицу измерения',
                  'Стоимость товаров (работ, услуг), имущественных прав без налога всего',
                  'Налоговая ставка',
                  'Стоимость товаров (работ, услуг), имущественных прав с налогом всего')

indexes: list[int] = []
# Удаляем последний список, так как в нем только сумма
cleaned_data.pop(-1)
print(f'Последний список --- {total_list}')

for lst in cleaned_data:
    for i in range(len(lst)):
        if lst[i] in needle_columns:
            indexes.append(i)

print(f'Индексы: {indexes}; Кол-во индексов: {len(indexes)}')

result = [[sublist[i] for i in indexes] for sublist in cleaned_data]

for i in range(2):
    result.pop(1)

print(result)

num: str = ''
for el in reversed(total_list):
    new_el = el.replace('.', '').replace(',', '').replace(' ', '')
    if new_el.isdigit():
        num = el

data['data_table'] = result
data['total'] = num

DictToJson.write_to_json(collection=data)


"""
наименование услуг ( табличная часть), количество, цена, налоговая ставка, сумма налога,стоимость работ или услуг,
а так же итоговую сумму налога и итоговую стоимость товаров(работ, услуг).
"""
