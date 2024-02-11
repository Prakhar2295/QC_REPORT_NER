import numpy as np
import torch
from transformers import BertForTokenClassification, BertTokenizerFast


MAX_LEN = 512
EPOCHS = 4
MODEL_PATH = 'Bert'
TOKENIZER = BertTokenizerFast('Bert/vocab.txt', lowercase=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#MAX_LEN = 512
EPOCHS = 4
DEVICE = torch.device("cpu")
#MODEL_PATH = 'Bert'
STATE_DICT = torch.load('Trained_NER_model/model_e10.tar', map_location=DEVICE)
TOKENIZER = BertTokenizerFast('Bert/vocab.txt', lowercase=True)
MODEL = BertForTokenClassification.from_pretrained(MODEL_PATH, state_dict=STATE_DICT['model_state_dict'], num_labels=9)

label_list = ["CERTIFICATE_NO","ISSUED_DATE","ORDER_NO","PURCHASE_ORDER","SUPPLIER_NAME","CUSTOMER_NAME","PLATE_NO","O","MISC"]
id2label = {k:v for k,v in enumerate(label_list)}
label2id = {v:k for k,v in enumerate(label_list)}

model = MODEL
model.to(DEVICE)


def tokenize_long_text(text):
	tok = TOKENIZER.encode_plus(text, add_special_tokens=False, return_tensors='pt', return_offsets_mapping=True)
	input_id_chunks = tok["input_ids"][0].split(510)
	mask_chunks = tok["attention_mask"][0].split(510)
	offset_mappings = tok["offset_mapping"][0].split(510)

	chunk_size = 512

	input_id_chunks = list(input_id_chunks)
	mask_chunks = list(mask_chunks)
	offset_mappings = list(offset_mappings)

	for i in range(len(input_id_chunks)):
		if len(input_id_chunks[i]) == chunk_size:
			pass
		else:
			input_id_chunks[i] = torch.cat([
				torch.tensor([101]), input_id_chunks[i], torch.tensor([102])
			])
		if len(mask_chunks[i]) == chunk_size:
			pass
		else:
			mask_chunks[i] = torch.cat([
				torch.tensor([1]), mask_chunks[i], torch.tensor([1])
			])
		if len(offset_mappings[i]) == chunk_size:
			pass
		else:
			offset_mappings[i] = torch.cat([
				torch.tensor([[0, 0]]), offset_mappings[i], torch.tensor([[0, 0]])
			])

		pad_len = chunk_size - input_id_chunks[i].shape[0]

		if pad_len > 0:
			input_id_chunks[i] = torch.cat([
				input_id_chunks[i], torch.tensor([0] * pad_len)
			])

			mask_chunks[i] = torch.cat([
				mask_chunks[i], torch.tensor([0] * pad_len)
			])

			offset_mappings[i] = torch.cat([

				offset_mappings[i], torch.tensor([[0, 0]] * pad_len)

			])

	input_ids = torch.stack(input_id_chunks)
	attention_mask = torch.stack(mask_chunks)
	offset_mapping = torch.stack(offset_mappings)
	input_dict = {

		"text": text,
		"input_ids": input_ids.long(),
		"attention_mask": attention_mask.long(),
		"offset_mapping": offset_mapping.long()

	}

	return input_dict



def model_inference_long_text(input_dict,text):
	model.eval()
	# data = process_resume2(test_resume, tokenizer, MAX_LEN)
	# input_ids, input_mask = data['input_ids'].unsqueeze(0), data['attention_mask'].unsqueeze(0)
	# @labels = torch.tensor([1] * input_ids.size(0), dtype=torch.long).unsqueeze(0)
	with torch.no_grad():
		outputs = model(
			input_dict["input_ids"],
			token_type_ids=None,
			attention_mask=input_dict["attention_mask"]
		)
		logits = outputs[0]

		final_entities = []
		entities = []
		for i in range(np.argmax(logits, axis=2).size(0)):
			for label_id, offset in zip(np.argmax(logits, axis=2)[i].tolist(),
			                            input_dict["offset_mapping"].numpy().tolist()[i]):
				# print(i)
				curr_id = id2label[label_id]
				curr_start = offset[0]
				curr_end = offset[1]
				if curr_id != 'O':
					if len(entities) > 0 and entities[-1]['entity'] == curr_id and curr_start - entities[-1]['end'] in [
						0, 1]:
						entities[-1]['end'] = curr_end
					else:
						entities.append({str(i): i, 'entity': curr_id, 'start': curr_start, 'end': curr_end})
			# final_entities.append(entities)

			for ent in entities:
				ent['text'] = text[ent['start']:ent['end']]
	return entities


#f = open("doc3_txt/doc_0.txt",'rb')
#text1 = f.read()
#text = str(text1)
#print(str(text))
#print(type(text))
##input_dict =tokenize_long_text(text)
#f =input_dict["input_ids"].size()
#print(f)
#entities = model_inference_long_text(input_dict,text)
#print(entities)














