import numpy
import transformers
from deeppavlov import train_model
from deeppavlov.core.commands.utils import parse_config
from deeppavlov import build_model, configs

# model = build_model(configs.squad.squad_ru_bert, download=True)
# model_config = parse_config('squad_bert')
model_config = parse_config('squad_ru_bert')

# model_config['dataset_reader']['data_path'] = r'dataset.json'
# model_config['dataset_reader']['url'] = ''
# model_config['dataset_reader']['dataset'] = 'test_train_set'
# model_config['chainer']['pipe'][3]['save_path'] = r'C:\Work\docs'
# model_config['metadata']['variables']['MODELS_PATH'] = r''
# model = train_model(model_config)
model = train_model(r"C:\Users\79118\PycharmProjects\DeepPavlov\.venv\Lib\site-packages\deeppavlov\configs\squad\squad_ru_bert.json")
model.save()

