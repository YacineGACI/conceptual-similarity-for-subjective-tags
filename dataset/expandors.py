import requests, pickle, re, json
import numpy as np
from scipy.stats import norm
from tqdm import tqdm
from nltk.corpus import wordnet as wn
import torch
from transformers import pipeline
from transformers import T5Tokenizer, T5Config, T5ForConditionalGeneration
from transformers import GPT2LMHeadModel, GPT2Tokenizer



class WordNetExpansion:
    def __init__(self, num_synsets, is_aspect=True, hyponym=True, meronym=True, hypernym=True, sisters=True):
        self.num_synsets = num_synsets
        self.is_aspect = is_aspect
        self.hyponym = hyponym
        self.meronym = meronym
        self.hypernym = hypernym
        self.sisters = sisters


    def keep_synset(self, syn, name, pos):
        if syn._pos != pos or syn._name.split('.')[0] != name:
            return False
        return True

    
    def format_word(self, word):
        return word.lower().replace("_", " ")


    def expand_one_word(self, word):
        expansions = set()

        for syn in wn.synsets(word)[:self.num_synsets]:
            if self.keep_synset(syn, word, 'n' if self.is_aspect else 's'):
                if self.hyponym:
                    try:
                        for h in syn.hyponyms():
                            expansions.add(self.format_word(h.name().split(".")[0]))
                    except:
                        pass

                if self.meronym:
                    try:
                        for h in syn.part_meronyms():
                            expansions.add(self.format_word(h.name().split(".")[0]))
                    except:
                        pass

                if self.hypernym:
                    try:
                        for h in syn.hypernyms():
                            expansions.add(self.format_word(h.name().split(".")[0]))
                    except:
                        pass

                if self.sisters:
                    try:
                        hypernym = syn.hypernyms()[0]
                        for s in hypernym.hyponyms():
                            expansions.add(self.format_word(s.name().split(".")[0]))
                    except:
                        pass

        return expansions


    def expand(self, words):
        expansions = set()
        for word in words:
            expansions = expansions | self.expand_one_word(word)
        return list(expansions - set(words))



    def expand_adjectives(self, words, capacity):
        expansions = set(words)

        for word in words:
            adjectives_synsets = [s for s in wn.synsets(word) if s._pos in ["a", "s"]]
            for syn in adjectives_synsets[:capacity]:

                for l in syn.lemmas():
                    expansions.add(self.format_word(l.name()))

                try:
                    for s in syn.similar_tos():
                        for l in s.lemmas():
                            expansions.add(self.format_word(l.name()))
                except:
                    pass


                try:
                    for s in syn.also_sees():
                        for l in s.lemmas():
                            expansions.add(self.format_word(l.name()))
                except:
                    pass

        return list(expansions)









class ConceptNetExpansion:
    def __init__(self, capacity, minimum_weight=2.0, second_level_expansion=True):
        self.capacity = capacity
        self.minimum_weight = minimum_weight
        self.second_level_expansion = second_level_expansion

    
    def format_word(self, word):
        return word[2:] if word.startswith("a ") else word


    def expand_one_word(self, word):
        expansions = set()

        # Get IsA relations
        isa_relations = requests.get("https://api.conceptnet.io/query?start=/c/en/{}&rel=/r/IsA&limit={}".format(word, self.capacity)).json()['edges']
        for isa in isa_relations:
            isa_term = isa["end"]["label"].lower()
            if isa["end"]["language"] == "en" and isa["weight"] > self.minimum_weight:
                expansions.add(self.format_word(isa_term))

                if self.second_level_expansion:
                    # Get all type of IsA aspects
                    typeof_isa_relations = requests.get("https://api.conceptnet.io/query?end=/c/en/{}&rel=/r/IsA&limit={}".format(isa_term.replace(" ", "_"), self.capacity)).json()['edges']
                    for typeof_isa in typeof_isa_relations:
                        typeof_isa_term = typeof_isa["start"]["label"].lower()
                        if typeof_isa["start"]["language"] == "en"  and typeof_isa["weight"] > self.minimum_weight:
                            expansions.add(self.format_word(typeof_isa_term))

        # Get Type of relations
        typeof_relations = requests.get("https://api.conceptnet.io/query?end=/c/en/{}&rel=/r/IsA&limit={}".format(word.replace(" ", "_"), self.capacity)).json()['edges']
        for typeof in typeof_relations:
            typeof_term = typeof["start"]["label"].lower()
            if typeof["start"]["language"] == "en" and typeof["weight"] > self.minimum_weight:
                expansions.add(self.format_word(typeof_term))

        return expansions


    def expand(self, words):
        expansions = set()
        for word in words:
            expansions = expansions | self.expand_one_word(word)
        return list(expansions - set(words))












class MaskedLMExpansion:
    def __init__(self, model, top_k=10):
        device = 0 if torch.cuda.is_available() else -1
        self.nlp = pipeline(task="fill-mask", model=model, tokenizer=model, device=device)
        self.top_k = top_k
        self.mask_token = "<mask>" if "roberta" in model else "[MASK]"

    
    def format_word(self, word):
        return word.strip(" ")


    def get_template(self, words):
        return ", ".join(words) + " and {} are all related concepts.".format(self.mask_token)


    def expand(self, words):
        template = self.get_template(words)
        output = self.nlp(template, top_k=self.top_k)
        expansions = []
        for o in output:
            if self.format_word(o["token_str"]) not in words:
                expansions.append(self.format_word(o["token_str"]))

        return expansions









class WordEmbeddingExpansion:
    def __init__(self, embeddings_filepath, num_words, distance_metric="euclidean"):
        with open(embeddings_filepath, 'rb') as f:
            self.model = pickle.load(f)
        self.num_words = num_words
        if distance_metric == "euclidean":
            self.distance = self.euclidean_distance
            self.comparison_operator = lambda x, y: x < y
        elif distance_metric == "cosine":
            self.distance = self.cosine_similarity
            self.comparison_operator = lambda x, y: x > y
        else:
            raise ValueError


    def euclidean_distance(self, a, b):
        a = np.array(a)
        b = np.array(b)
        return np.linalg.norm(a - b)

    
    def cosine_similarity(self, a, b):
        a = np.array(a)
        b = np.array(b)
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


    
    def format_word(self, word):
        return word.replace("_", " ")


    def insert_top_values(self, list, new_word, new_distance):
        # Find the index where to put new value
        index = 0
        while index < self.num_words and list[index] is not None and self.comparison_operator(list[index][1], new_distance):
            index += 1
        
        if index == self.num_words:
            # new_distance is too large. Do not insert 
            pass
        elif list[index] is None:
            # The list is not fully populated. Insert the new value
            list[index] = (new_word, new_distance)
        else:
            transition_index = self.num_words - 1
            while transition_index > index:
                list[transition_index] = list[transition_index - 1]
                transition_index -= 1
            list[index] = (new_word, new_distance)
        return list


        

    def expand(self, words):
        expansions = [None] * self.num_words

        for vocab_word in self.model.keys():
            if vocab_word not in words:
                total_distance = 0
                for word in words:
                    try:
                        total_distance += self.distance(self.model[word], self.model[vocab_word])
                    except:
                        pass # This is when the seed does not belong to the vocabulary of the embedding model
                
                expansions = self.insert_top_values(expansions, vocab_word, total_distance)
        
        expansions = [self.format_word(x[0]) for x in expansions]
        return expansions










class T5Expansion:
    def __init__(self, t5_path, num_return_sequences, max_length, num_beams=200):
        self.num_return_sequences = num_return_sequences
        self.max_length = max_length
        self.num_beams = num_beams

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # My envirnment uses CPU
        self.tokenizer = T5Tokenizer.from_pretrained(t5_path)
        config = T5Config.from_pretrained(t5_path)
        self.model = T5ForConditionalGeneration.from_pretrained(t5_path, config=config).to(self.device)

    
    def get_template(self, words):
        return ", ".join(words) + " and <extra_id_0> are all related concepts."


    def expand(self, words):
        text = self.get_template(words)
        encoded = self.tokenizer.encode_plus(text, add_special_tokens=True, return_tensors='pt')
        input_ids = encoded['input_ids'].to(self.device)
        outputs = self.model.generate(input_ids=input_ids, num_beams=self.num_beams, num_return_sequences=self.num_return_sequences, max_length=self.max_length)
        
        _0_index = text.index('<extra_id_0>')
        _result_prefix = text[:_0_index]
        _result_suffix = text[_0_index+12:]  # 12 is the length of <extra_id_0>

        def _filter(output, end_token='<extra_id_1>'):
            # The first token is <unk> (inidex at 0) and the second token is <extra_id_0> (indexed at 32099)
            _txt = self.tokenizer.decode(output[2:], skip_special_tokens=False, clean_up_tokenization_spaces=False)
            if end_token in _txt:
                _end_token_index = _txt.index(end_token)
                return _result_prefix + _txt[:_end_token_index] + _result_suffix
            else:
                return _result_prefix + _txt + _result_suffix

        results = list(map(_filter, outputs))
        regex_template = text.replace("<extra_id_0>", "(.*)")

        expansions = []
        for r in results:
            e = re.findall(regex_template, r)
            if e[0] not in words:
                expansions.append(e[0])

        return expansions








class TextGenerationExpansion:
    def __init__(self, model, num_return_sequences, max_length, num_beams=200):
        self.num_return_sequences = num_return_sequences
        self.max_length = max_length
        self.num_beams = num_beams

        self.tokenizer = GPT2Tokenizer.from_pretrained(model)
        self.model = GPT2LMHeadModel.from_pretrained(model, pad_token_id=self.tokenizer.eos_token_id)


    def get_template(self, words):
        return "These concepts are related: " + ", ".join(words) + " and"


    def format_word(self, word):
        return word.strip(" \n.,§!?;:")

    
    def expand(self, words):
        text = self.get_template(words)
        input_ids = self.tokenizer.encode(text, return_tensors='pt')
        beam_output = self.model.generate(
            input_ids, 
            max_length=len(input_ids[0]) + self.max_length, 
            num_beams=self.num_beams,
            num_return_sequences=self.num_return_sequences,
            early_stopping=True
        )

        expansions = set()
        for i in range(len(beam_output)):
            decoded_sentence = self.tokenizer.decode(beam_output[i], skip_special_tokens=True)
            new_word = self.format_word(decoded_sentence.split(text, 1)[1])
            if new_word not in words:
                expansions.add(new_word)

        return list(expansions)












class AspectTermExpansion:
    def __init__(self, config_filepath, min_consensus_rate=0.5):
        self.config_filepath = config_filepath
        self.min_consensus_rate = min_consensus_rate


    def expand(self, words):
        with open(self.config_filepath, 'r') as f:
            configs = json.load(f)["aspect"]

        all_expansions = []
        all_expanded_words = set()

        for config in tqdm(configs):
            # Instantiate the expendor
            expandor = globals()[config["name"]](**config["params"])

            expansions = expandor.expand(words)
            all_expansions.append(expansions)
            all_expanded_words = all_expanded_words | set(expansions)

            del expandor

        output = []
        for word in all_expanded_words:
            num_votes = 0
            for e in all_expansions:
                if word in e:
                    num_votes += 1

            if num_votes / len(configs) >= self.min_consensus_rate:
                output.append(word)

        return output








class OpinionTermExpansion:
    def __init__(self, config_filepath, capacity=3, filtering_strategy="both", mlm="bert-base-uncased", lm="gpt2", std_factor=1.0):
       
        with open(config_filepath, 'r') as f:
            configs = json.load(f)["opinion"]

        self.capacity = configs["capacity"] if "capacity" in configs.keys() else capacity
        self.filtering_strategy = configs["filtering_strategy"] if "filtering_strategy" in configs.keys() else filtering_strategy
        self.std_factor = configs["std_factor"] if "std_factor" in configs.keys() else std_factor
        lm = configs["lm"] if "lm" in configs.keys() else lm
        mlm = configs["mlm"] if "mlm" in configs.keys() else mlm

        if self.filtering_strategy in ["lm", "both"]:
            self.lm_tokenizer = GPT2Tokenizer.from_pretrained(lm)
            self.lm = GPT2LMHeadModel.from_pretrained(lm, pad_token_id=self.lm_tokenizer.eos_token_id)
            self.lm.eval()
        if self.filtering_strategy in ["mlm", "both"]:
            device = 0 if torch.cuda.is_available() else -1
            self.mlm = pipeline(task="fill-mask", model=mlm, tokenizer=mlm, device=device)


    def format_word(self, word):
        return word.lower().replace("_", " ")

    
    def generate_templates(self, aspect, opinion):
        templates = [
            "The {0} is {1}.",
            "That {0} is {1}.",
            "That {0} is really {1}.",
            "{1} {0}.",
            "Really {1} {0}."
        ]
        return [t.format(aspect, opinion) for t in templates]
    

    def lm_loss(self, aspects, opinion):
        sentences = []
        for a in aspects:
            sentences += self.generate_templates(a, opinion)
        total_loss = 0
        for s in sentences:
            input_ids = torch.tensor(self.lm_tokenizer.encode(s, add_special_tokens=True)).unsqueeze(0)  # Batch size 1
            outputs = self.lm(input_ids, labels=input_ids)
            loss = outputs[0]
            total_loss += loss.item()
        return (total_loss / len(sentences))


    def mlm_loss(self, aspects, opinion):
        sentences = self.generate_templates("[MASK]", opinion)
        total_prob = 0
        for s in sentences:
            output = self.mlm(s, targets=aspects)
            for o in output:
                total_prob += o["score"]
        return (total_prob / (len(sentences) * len(aspects)))


    def expand(self, words, aspects=None):
        expansions = set(words)

        # 1/ Get all possible expansions from WordNet
        for word in words:
            adjectives_synsets = [s for s in wn.synsets(word) if s._pos in ["a", "s"]]
            for syn in adjectives_synsets[:self.capacity]:

                for l in syn.lemmas():
                    expansions.add(self.format_word(l.name()))

                try:
                    for s in syn.similar_tos():
                        for l in s.lemmas():
                            expansions.add(self.format_word(l.name()))
                except:
                    pass


                try:
                    for s in syn.also_sees():
                        for l in s.lemmas():
                            expansions.add(self.format_word(l.name()))
                except:
                    pass

            
        # 2/ Filter out unwanted expansions
        if aspects is None:
            return list(expansions)
        
        else:
            filtered_result = words

            for strategy, loss_fct in [("mlm", self.mlm_loss), ("lm", self.lm_loss)]:
                if self.filtering_strategy in [strategy, "both"]:
                    seed_scores = []
                    for w in words:
                        score = loss_fct(aspects, w)
                        seed_scores.append(score)
                    
                    # Estimate Normal distribution
                    mu, std = norm.fit(seed_scores)
                    
                    # Filter
                    for o in expansions:
                        if o not in filtered_result:
                            score = loss_fct(aspects, o)
                            if score < mu + self.std_factor * std and score > mu - self.std_factor * std:
                                filtered_result.append(o)

            return filtered_result