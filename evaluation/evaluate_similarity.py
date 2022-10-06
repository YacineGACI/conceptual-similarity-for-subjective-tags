from os import listdir
from os.path import isfile, join
import argparse

from sklearn.metrics import mean_absolute_error
from scipy.stats import pearsonr, spearmanr
import torch 

from models import Conceptual_Similarity




def get_files_in_folder(folder):
    return [join(folder, f) for f in listdir(folder) if isfile(join(folder, f))]


def swap_aspect_opinion(tag):
    quantifiers = ["super", "highly", "really", "walking", "perfectly", "pretty", "insanely", "brilliantly", "very", "extremely", "well", "stunningly", "just", "quite", "over", "so", "nicely", "pleasantly", "consistently", "poorly"]
    words = tag.split(" ")
    if len(words) == 2:
        return words[1] + " " + words[0]
    elif len(words) == 3:
        if words[1] in quantifiers:
            return words[1] + " " + words[2] + " " + words[0]
        else:
            return words[2] + " " + words[0] + " " + words[1]
    else:
        return words[2] + " " + words[3] + " " + words[0] + " " + words[1]



def read_eval_data(filename):
    sentences, labels = [], []
    with open(filename, 'r') as f:
        for line in f.readlines():
            s1, s2, l = line.strip('\n ').split(',')

            s1 = swap_aspect_opinion(s1)
            s2 = swap_aspect_opinion(s2)

            sentences.append((s1, s2))
            labels.append(int(l))

    return sentences, labels



def merge_labels(labels):
    new_labels = []
    for i in range(len(labels[0])):
        tmp = []
        for l in labels:
            tmp.append(l[i])
        merged_value = sum(tmp) / len(tmp)
        new_labels.append(merged_value)
    return new_labels







def compute_similarity(s1, s2):
    tmp = tokenizer.encode_plus(s1, s2, add_special_tokens=True, padding="max_length", max_length=20)
    input_ids = torch.tensor(tmp['input_ids']).unsqueeze(0)
    attention_mask = torch.tensor(tmp['attention_mask']).unsqueeze(0)
    seg = torch.tensor(tmp['token_type_ids']).unsqueeze(0)
    h_0 = torch.rand(2, input_ids.shape[0], hyperparameters['lstm_hidden_size'])
    c_0 = torch.rand(2, input_ids.shape[0], hyperparameters['lstm_hidden_size'])

    output = model(input_ids, attention_mask, seg, h_0, c_0)
    return output.item()






def merge_predictions(predictions, weights):
    new_predictions = []
    for i in range(len(predictions[0])):
        mean_pred = 0
        for p, w in zip(predictions, weights):
            mean_pred += p[i] * w
        mean_pred /= sum(weights)
        new_predictions.append(mean_pred)
    return new_predictions




if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    
    parser.add_argument("--data", type=str, required=True, help="Path to evaluation folder")
    parser.add_argument("--model", type=str, required=True, help="Path to similarity model")
    args = parser.parse_args()

    # Find all evaluation files
    eval_directory = args.data
    eval_files = get_files_in_folder(eval_directory)
    
    # Load corresponding evaluation data
    data = [read_eval_data(f) for f in eval_files]
    sentences = data[0][0]
    labels = [x[1] for x in data]

    # Merge the labels from different workers
    labels = [x/5 for x in merge_labels(labels)]
    
    # Load the similarity model
    saved = torch.load(args.model)
    hyperparameters = saved["hyperparameters"]
    model = Conceptual_Similarity(bert_hidden_size=hyperparameters['bert_hidden_size'], lstm_hidden_size=hyperparameters['lstm_hidden_size'], classifier_hidden_size=hyperparameters['classifier_hidden_size'], dropout=hyperparameters['dropout'], pooling=hyperparameters['pooling'])
    model.load_state_dict(saved['model'])
    model.eval()
    tokenizer = model.tokenizer

    predictions = []
    for s1, s2 in sentences:
        predictions.append(compute_similarity(s1, s2))

    assert len(predictions) == len(labels)

    print("Pearson correlation: {}".format(pearsonr(predictions, labels)[0]))
    print("Spearman correlation: {}".format(spearmanr(predictions, labels)[0]))
    print("Mean Absolute Error: {}".format(mean_absolute_error(predictions, labels)))