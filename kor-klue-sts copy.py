#%%
import math
import logging
from datetime import datetime

import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from sentence_transformers import SentenceTransformer, models, LoggingHandler, losses, util
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers.readers import InputExample
import pandas as pd
import warnings

warnings.filterwarnings('ignore')
num_epoch = 4
batch_size = 32
max_len = 64
data_name='inference'
mode='train'
pretrained_model_name = 'klue/roberta-base'
device = torch.device("cuda")
sts_model_save_path = 'output/training_'+data_name+'-'+pretrained_model_name.replace("/", "-")+'-'+datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# %%# logger
logging.basicConfig(
    format="%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    handlers=[LoggingHandler()],
)
#%%
def load_data(name):
    path = '/home/iosys/RJH/Project/kluests/data/dataset'
    columns = ['sentence1','sentence2','score']
    #PATH = '/home/iosys/RJH/Project/korSTS/data/dataset'
    data = pd.read_csv(path+f'/sts-{name}.tsv', delimiter='\t', on_bad_lines='skip')[columns]

    sent2_null_idx = data[data['sentence2'].isnull()].index

    if len(sent2_null_idx)>1: 
        for i in sent2_null_idx:
            pair = data['sentence1'].iloc[i].split(sep='\t')
            data['sentence1'].iloc[i] = pair[0]
            data['sentence2'].iloc[i] = pair[1]
    
    data.to_csv(path+f'/sts-{name}.csv')
    return data

# %%
def make_sts_input_example(dataset):
    ''' 
    Transform to InputExample
    ''' 
    input_examples = []
    for i, data in enumerate(dataset):
        sentence1 = data['sentence1']
        sentence2 = data['sentence2']
        try:
            score = (data['labels']['label']) / 5.0  # normalize 0 to 5
        except:
            score = (data['score']) / 5.0
            
        input_examples.append(InputExample(texts=[sentence1, sentence2], label=score))

    return input_examples


# %%
# Train Dataloader

def dataloader_train(data):
    train_dataloader = DataLoader(
        data,
        shuffle=True,
        batch_size=batch_size,
    )
    return train_dataloader

def evaluator_test(data, data_name):
    dev_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(
        data,
        name=data_name,
    )
    return dev_evaluator

# %%
# Load Embedding Model
def model_module(pretrained_model_name):
    embedding_model = models.Transformer(
        model_name_or_path=pretrained_model_name, 
        max_seq_length=max_len,
        do_lower_case=True
    )

    # Only use Mean Pooling -> Pooling all token embedding vectors of sentence.
    pooling_model = models.Pooling(
        embedding_model.get_word_embedding_dimension(),
        pooling_mode_mean_tokens=True,
        pooling_mode_cls_token=False,
        pooling_mode_max_tokens=False,
    )
    return embedding_model, pooling_model

#%%
embedding_model, pooling_model = model_module(pretrained_model_name)
model = SentenceTransformer(modules=[embedding_model, pooling_model])
checkpoint = torch.load('/home/iosys/RJH/Project/kluests/output/training_kor_klue_sts-klue-roberta-base-2024-02-01_17-38-09/model.pt')
model.load_state_dict(checkpoint)
#%%

sent1 = '삼성라이온즈는 1982년에 창단되어 롯데자이언츠와 함께 단 한 번도 모기업이 바뀌지 않은 야구단 입니다.'#input()
sent2 = '삼성라이온즈는 모기업을 삼성으로 한 야구단으로, 1982년에 창단되었습니다.'#input()
score = round(3, 2)#round(float(input()),2)

pd.DataFrame({'sentence1':[sent1],
              'sentence2': [sent2], 
              'score':[score]}).to_csv('/home/iosys/RJH/Project/kluests/data/dataset/sts-inference.csv')

inference_data = load_dataset("csv", data_files="/home/iosys/RJH/Project/kluests/data/dataset/sts-inference.csv")['train']
inference_data_examples = make_sts_input_example(inference_data)
test_evaluator = evaluator_test(inference_data_examples, data_name)
result = test_evaluator(model, output_path=sts_model_save_path)[0]*5
result = round(result, 2)
error = round(abs(result-score),2)
print('----')
print('측정된 두 문장의 유사도는 {0}점이며, 실제 점수와 {1}점 차이 입니다. \n실제 유사도는 {2}점입니다.'.format(result, error, score))
#%%
# %%
