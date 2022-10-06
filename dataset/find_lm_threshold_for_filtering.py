import random, tqdm, random
import torch

def to_template(aspect, opinion):
    return "The {} is {}".format(aspect, opinion)

def read_seed_words(filename):
    concepts = {}
    with open(filename, 'r') as f:
        text = f.read()
        atts = text.strip(" \n\t\r").split('\n\n')
        for a in atts:
            lines = a.split('\n')
            concepts[lines[0]] = [[w.strip(' ') for w in l.split(',')] for l in lines[1:]]
    return concepts


def construct_templates(seeds, num_op_per_as=3):
    templates = []

    for k in seeds.keys():
        aspects = seeds[k][0]
        opinions = seeds[k][1:]

        for a in aspects:
            num_opinion_lists = 0
            # Postive examples
            for op in opinions:
                num_opinion_lists += 1
                for _ in range(num_op_per_as):
                    o = random.choice(op)
                    templates.append("The {} is {}".format(a, o))

            # Negative examples
            # Select an aspect
            while True:
                random_aspect = random.choice(list(seeds.keys()))
                if random_aspect != k:
                    break

            for _ in range(num_opinion_lists):
                op = random.randint(0, len(seeds[random_aspect]) - 2) # -1 because the first element of this list is always for aspects, and another - 1 because randint includes the values of the range
                for _ in range(num_op_per_as):
                    o = random.choice(seeds[random_aspect][1:][op])
                    templates.append(to_template(a, o))
    
    return templates


def compute_likelihoods(templates,tokenizer, lm_model):
    likelihoods = []
    for t in tqdm.tqdm(templates):
        input_ids = torch.tensor(tokenizer.encode(t, add_special_tokens=True)).unsqueeze(0)  # Batch size 1
        outputs = lm_model(input_ids, labels=input_ids)
        loss = outputs[0]
        likelihoods.append(loss.item())
    return likelihoods



def compute_threshold(seeds, tokenizer, lm_model):
    templates = construct_templates(seeds)
    likelihoods = compute_likelihoods(templates, tokenizer, lm_model)
    return sum(likelihoods) / len(likelihoods)
