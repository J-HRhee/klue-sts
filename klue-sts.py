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

# %%# logger
logging.basicConfig(
    format="%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    handlers=[LoggingHandler()],
)
pretrained_model_name = 'klue/roberta-base'
sts_num_epochs = 4
train_batch_size = 32
device = torch.device("cuda")

#sts_model_save_path = 'output/training_sts-'+pretrained_model_name.replace("/", "-")+'-'+datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
sts_model_save_path = '/home/iosys/RJH/Project/kluests/output/training_sts-klue-roberta-base-2024-02-01_14-42-45'
# %%
klue_sts_train = load_dataset("klue", "sts", split='train[:90%]')
klue_sts_valid = load_dataset("klue", "sts", split='train[-10%:]') # train의 10%를 validation set으로 사용
klue_sts_test = load_dataset("klue", "sts", split='validation')
korsts_test = load_dataset("csv", data_files="/home/iosys/RJH/Project/kluests/data/dataset/sts-test.csv")['train']


print('Length of Train : ',len(klue_sts_train))
print('Length of Valid : ',len(klue_sts_valid))
print('Length of Test : ',len(klue_sts_test))
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

sts_train_examples = make_sts_input_example(klue_sts_train)
sts_valid_examples = make_sts_input_example(klue_sts_valid)
sts_test_examples = make_sts_input_example(klue_sts_test)
korsts_test_example = make_sts_input_example(korsts_test)
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
korsts_test_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(
    korsts_test_example,
    name="korsts-test",
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
# Use CosineSimilarityLoss
try:
    checkpoint = torch.load(sts_model_save_path+"/model.pt")
    #model2 = SentenceTransformer(modules=[embedding_model, pooling_model])
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

#%%

test_evaluator(model, output_path=sts_model_save_path)
# %%
korsts_test_evaluator(model, output_path=sts_model_save_path)

# %%
