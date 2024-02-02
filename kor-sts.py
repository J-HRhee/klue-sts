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
batch_size = 64
data_name='korsts'
mode='train'

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

for name in ['train','dev','test']:
    load_data(name)
# %%# logger
logging.basicConfig(
    format="%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    handlers=[LoggingHandler()],
)
pretrained_model_name = 'klue/roberta-base'
sts_num_epochs = num_epoch
train_batch_size = batch_size
device = torch.device("cuda")

#sts_model_save_path = 'output/training_sts-'+pretrained_model_name.replace("/", "-")+'-'+datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
#sts_model_save_path = '/home/iosys/RJH/Project/kluests/output/training_sts-klue-roberta-base-2024-02-01_14-42-45'
sts_model_save_path = 'output/training_korsts-'+pretrained_model_name.replace("/", "-")+'-'+datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
# %%
korsts_train= load_dataset("csv", data_files="/home/iosys/RJH/Project/kluests/data/dataset/sts-train.csv")['train']
korsts_valid = load_dataset("csv", data_files="/home/iosys/RJH/Project/kluests/data/dataset/sts-dev.csv")['train']
korsts_test = load_dataset("csv", data_files="/home/iosys/RJH/Project/kluests/data/dataset/sts-test.csv")['train']
klue_sts_test = load_dataset("klue", "sts", split='validation')

print('Length of Train : ',len(korsts_train))
print('Length of Valid : ',len(korsts_valid))
print('Length of Test : ',len(korsts_test))
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

sts_train_examples = make_sts_input_example(korsts_train)
sts_valid_examples = make_sts_input_example(korsts_valid)
sts_test_examples = make_sts_input_example(korsts_test)
klue_sts_test_examples = make_sts_input_example(klue_sts_test)
# %%
# Train Dataloader
train_dataloader = DataLoader(
    sts_train_examples,
    shuffle=True,
    batch_size=train_batch_size,
)

# Evaluator by sts-validation
dev_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(
    sts_valid_examples,
    name="sts-dev",
)

# Evaluator by sts-test
test_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(
    sts_test_examples,
    name="sts-test",
)

# Evaluator by korsts-test
klue_test_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(
    klue_sts_test_examples,
    name="klue-test",
)
# %%
# Load Embedding Model
embedding_model = models.Transformer(
    model_name_or_path=pretrained_model_name, 
    max_seq_length=32,
    do_lower_case=True
)

# Only use Mean Pooling -> Pooling all token embedding vectors of sentence.
pooling_model = models.Pooling(
    embedding_model.get_word_embedding_dimension(),
    pooling_mode_mean_tokens=True,
    pooling_mode_cls_token=False,
    pooling_mode_max_tokens=False,
)

model = SentenceTransformer(modules=[embedding_model, pooling_model])
# %%
import os

file_list = os.listdir('/home/iosys/RJH/Project/kluests/output')
file_list
#%%
# Use CosineSimilarityLoss
if mode != 'train':
    try:
        file_list = os.listdir('/home/iosys/RJH/Project/kluests/output')
        for file in file_list:
            if data_name not in file:
                sts_model_save_path="/home/iosys/RJH/Project/kluests/output/"+file
                checkpoint = torch.load(sts_model_save_path+'/model.pt')
                model.load_state_dict(checkpoint)

    except:
        train_loss = losses.CosineSimilarityLoss(model=model)

        # warmup steps
        warmup_steps = math.ceil(len(sts_train_examples) * sts_num_epochs / train_batch_size * 0.1) #10% of train data for warm-up
        logging.info("Warmup-steps: {}".format(warmup_steps))

        model.to(device)

        # Training
        model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            evaluator=dev_evaluator,
            epochs=sts_num_epochs,
            evaluation_steps=int(len(train_dataloader)*0.1),
            warmup_steps=warmup_steps,
            output_path=sts_model_save_path
        )

        torch.save(model.state_dict(), sts_model_save_path+'/model.pt')
else:
    train_loss = losses.CosineSimilarityLoss(model=model)

    # warmup steps
    warmup_steps = math.ceil(len(sts_train_examples) * sts_num_epochs / train_batch_size * 0.1) #10% of train data for warm-up
    logging.info("Warmup-steps: {}".format(warmup_steps))

    model.to(device)

    # Training
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        evaluator=dev_evaluator,
        epochs=sts_num_epochs,
        evaluation_steps=int(len(train_dataloader)*0.1),
        warmup_steps=warmup_steps,
        output_path=sts_model_save_path
    )
    torch.save(model.state_dict(), sts_model_save_path+'/model.pt')

    
#%%
test_evaluator(model, output_path=sts_model_save_path)
#%%
klue_test_evaluator(model, output_path=sts_model_save_path)
# %%
