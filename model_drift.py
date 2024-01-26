import pandas as pd
from deepchecks.nlp import TextData
from data_preparation import data_preparation
from transformers import BertForTokenClassification, BertTokenizerFast
from deepchecks.nlp.checks import TrainTestPerformance
import torch
import ast
import numpy as np


def get_label(offset, labels):
    if offset[0] == 0 and offset[1] == 0:
        return 'O'
    for label in labels:
        if offset[1] >= label[0] and offset[0] <= label[1]:
            return label[2]
    return 'O'


def process_data(data, tokenizer, tag2idx, max_len, is_test=False):
	tok = tokenizer.encode_plus(data[0], max_length=max_len, return_offsets_mapping=True)
	curr_sent = {'orig_labels': [], 'labels': []}

	padding_length = max_len - len(tok['input_ids'])
	final = []
	if not is_test:
		labels = data[1]['entities']
		labels.reverse()
		for off in tok['offset_mapping']:
			label = data_preparation.get_label(off, labels)
			curr_sent['orig_labels'].append(label)
			curr_sent['labels'].append(tag2idx[label])
		curr_sent['labels'] = curr_sent['labels'] + ([0] * padding_length)

	curr_sent['input_ids'] = tok['input_ids'] + ([0] * padding_length)
	curr_sent['token_type_ids'] = tok['token_type_ids'] + ([0] * padding_length)
	curr_sent['attention_mask'] = tok['attention_mask'] + ([0] * padding_length)

	final.append({"text": tok.tokens(), "labels": curr_sent['orig_labels']})
	return curr_sent, final

def train_data_tokens(data, tokenizer, label2id):
    final_train_data = []
    for i in range(30):
        _, f = process_data(data[i], tokenizer, label2id, 512, is_test=False)
        final_train_data.append(f)
    return final_train_data


def train_text_labels(final_train_data:list):
	train_text = []
	train_labels = []
	for item in final_train_data:
		# text.append(item["text"])
		# print(item[0]["labels"])
		train_text.append(item[0]["text"])
		# print(item[0]["labels"])
		train_labels.append(item[0]["labels"])
	return train_text,train_labels

def test_data_tokens(data, tokenizer, label2id):
	final_test_data = []
	for i in range(400,430,1):
		_, f = process_data(data[i], tokenizer, label2id, 512, is_test=False)
		final_test_data.append(f)
	return final_test_data

def test_text_labels(final_test_data:list):
	test_text = []
	test_labels = []
	for item in final_test_data:
		# text.append(item["text"])
		# print(item[0]["labels"])
		test_text.append(item[0]["text"])
		# print(item[0]["labels"])
		test_labels.append(item[0]["labels"])
	return test_text,test_labels


MAX_LEN = 512
EPOCHS = 4
DEVICE = torch.device("cpu")
MODEL_PATH = '../input/bert-base-uncased'
STATE_DICT = torch.load('/kaggle/input/trained-ner-model/model_e10.tar', map_location=DEVICE)
TOKENIZER = BertTokenizerFast('../input/bert-base-uncased/vocab.txt', lowercase=True)
MODEL = BertForTokenClassification.from_pretrained(MODEL_PATH, state_dict=STATE_DICT['model_state_dict'], num_labels=9)


label_list = ["CERTIFICATE_NO","ISSUED_DATE","ORDER_NO","PURCHASE_ORDER","SUPPLIER_NAME","CUSTOMER_NAME","PLATE_NO","O","MISC"]
id2label = {k:v for k,v in enumerate(label_list)}
label2id = {v:k for k,v in enumerate(label_list)}

model = MODEL
model.to(DEVICE)

###Predicted labels

def process_prediction_data(data, tokenizer, max_len):
	tok = tokenizer.encode_plus(data, max_length=max_len, return_offsets_mapping=True)

	curr_sent = dict()

	padding_length = max_len - len(tok['input_ids'])

	curr_sent['input_ids'] = tok['input_ids'] + ([0] * padding_length)
	curr_sent['token_type_ids'] = tok['token_type_ids'] + ([0] * padding_length)
	curr_sent['attention_mask'] = tok['attention_mask'] + ([0] * padding_length)

	final_data = {
		'input_ids': torch.tensor(curr_sent['input_ids'], dtype=torch.long),
		'token_type_ids': torch.tensor(curr_sent['token_type_ids'], dtype=torch.long),
		'attention_mask': torch.tensor(curr_sent['attention_mask'], dtype=torch.long),
		'offset_mapping': tok['offset_mapping'],
		'tokens': tok.tokens()
	}

	return final_data


def predict(model, tokenizer, idx2tag, tag2idx, device, data):
	model.eval()
	data = process_prediction_data(data, tokenizer, MAX_LEN)
	# print(type(data["offset_mapping"]))
	# print(data["offset_mapping"])
	input_ids, input_mask = data['input_ids'].unsqueeze(0), data['attention_mask'].unsqueeze(0)
	labels = torch.tensor([1] * input_ids.size(0), dtype=torch.long).unsqueeze(0)
	with torch.no_grad():
		outputs = model(
			input_ids,
			token_type_ids=None,
			attention_mask=input_mask,
			labels=labels,
		)
		tmp_eval_loss, logits = outputs[:2]

	logits = logits.cpu().detach().numpy()
	# print(logits)
	label_ids = np.argmax(logits, axis=2)
	# print(type(label_ids))
	# print(label_ids)
	# print(len(label_ids[0]))

	final_data = []
	entities = []
	for label_id, offset in zip(label_ids[0], data['offset_mapping']):
		curr_id = idx2tag[label_id]
		curr_start = offset[0]
		curr_end = offset[1]
		if curr_id != 'O':
			if len(entities) > 0 and entities[-1]['entity'] == curr_id and curr_start - entities[-1]['end'] in [0, 1]:
				entities[-1]['end'] = curr_end
			else:
				entities.append({'entity': curr_id, 'start': curr_start, 'end': curr_end})
	final_data.append([data["tokens"], entities, data])
	for ent in entities:
		ent['text'] = data[ent['start']:ent['end']]
	return entities, final_data


def test_data_prediction(data, model, tokenizer, id2label, label2id, device):
    final_test_pred = []
    for i in range(400, 430, 1):
        _, f = predict(model, tokenizer, id2label, label2id, device, data[i][0])
        final_test_pred.append(f)
    return final_test_pred


def train_data_prediction(data, model, tokenizer, id2label, label2id, device):
	final_train_pred = []
	for i in range(400, 430, 1):
		_, f = predict(model, tokenizer, id2label, label2id, device, data[i][0])
		final_train_pred.append(f)
	return final_train_pred


def data_pred_for_drfit(final_list, TOKENIZER, get_label, label2id):
    data_list = []

    for document in final_list:
        text = document[0][2]
        tok = TOKENIZER.encode_plus(text, max_length=512, return_offsets_mapping=True)
        curr_sent = {'orig_labels': [], 'labels': []}

        padding_length = 512 - len(tok['input_ids'])

        is_test = False
        if not is_test:
            labels = []
            for entity_info in document[0][1]:
                #print(entity_info["start"], entity_info["end"], entity_info["entity"])
                labels.append([entity_info["start"], entity_info["end"], entity_info["entity"]])
            labels.reverse()
            #print(labels)
            for off in tok["offset_mapping"]:
                label = get_label(off, labels)
                curr_sent['orig_labels'].append(label)
                curr_sent['labels'].append(label2id[label])
            curr_sent['labels'] = curr_sent['labels'] + ([0] * padding_length)

        # Uncomment these lines if you need to include tokenized input information
        # curr_sent['input_ids'] = tok['input_ids'] + ([0] * padding_length)
        # curr_sent['token_type_ids'] = tok['token_type_ids'] + ([0] * padding_length)
        # curr_sent['attention_mask'] = tok['attention_mask'] + ([0] * padding_length)

        data_list.append({"tokens": tok.tokens(), "labels": curr_sent['orig_labels']})

    return data_list

final_test = test_data_prediction(data, model, tokenizer, id2label, label2id, device)
test_list=data_pred_for_drfit(final_test, TOKENIZER, get_label, label2id)

train_list=train_data_pred(final_train, TOKENIZER, get_label, label2id)



df_test_pred = pd.DataFrame(test_list)

df_train_pred = pd.DataFrame(train_list)

###For test data

def test_pred_drift(df_test_pred):
	test_data_text_pred = []
	test_data_labels_pred = []
	#df_test_pred = pd.read_csv("/content/test_pred.csv")
	for i in range(len(df_test_pred)):
	  #print(i)
	  output_text = ast.literal_eval(df_test_pred["tokens"][i])
	  output_labels = ast.literal_eval(df_test_pred["labels"][i])
	  test_data_text_pred.append(output_text)
	  test_data_labels_pred.append(output_labels)
	return test_data_text_pred,test_data_labels_pred

def train_pred_drift(df_train_pred):
	train_data_text_pred = []
	train_data_labels_pred = []
	#df_train_pred = pd.read_csv("/content/train_pred.csv")
	for i in range(len(df_train_pred)):
		# print(i)
		output_text = ast.literal_eval(df_train_pred["tokens"][i])
		output_labels = ast.literal_eval(df_train_pred["labels"][i])
		train_data_text_pred.append(output_text)
		train_data_labels_pred.append(output_labels)
	return train_data_text_pred,train_data_labels_pred





###Creating the text data object fro deepchecks
train = TextData(tokenized_text=train_data_text, label=train_data_labels, task_type='token_classification')
test = TextData(tokenized_text=test_data_text, label=test_data_labels, task_type='token_classification')

check = TrainTestPerformance().add_condition_train_test_relative_degradation_less_than()
result = check.run(train, test, train_predictions=train_data_labels_pred, test_predictions=test_data_labels_pred)

result.to_json()







