import itertools, random, math, argparse, tqdm
from dataset.expandors import AspectTermExpansion, OpinionTermExpansion



def read_seed_words(filename):
    concepts = {}
    with open(filename, 'r') as f:
        text = f.read()
        atts = text.strip(" \n\t\r").split('\n\n')
        for a in atts:
            lines = a.split('\n')
            concepts[lines[0]] = {
                "aspects": [w.strip(' ') for w in lines[1].split(',')],
                "opinions": [[w.strip(' ') for w in l.split(',')] for l in lines[2:]]
            }
            
    return concepts




def generate_new_dataset(seeds, expansion_config_file, output_filename, size, min_ratio_pos, as_min_consensus_rate):
    # Step 1: Expand seed words
    print("##############################")
    print("<<<  Expanding seed words  >>>")
    print("##############################")

    aspect_expendor = AspectTermExpansion(expansion_config_file, min_consensus_rate=as_min_consensus_rate)
    opinion_expendor = OpinionTermExpansion(expansion_config_file)

    concepts = list(seeds.keys())
    data = {c: {} for c in concepts}

    for concept in concepts:
        print(">>> Treating the concept of: ", concept)

        # Expend aspect terms
        data[concept]["aspects"] = aspect_expendor.expand(seeds[concept]["aspects"])

        # Expend opinions
        data[concept]["opinions"] = []
        for o in seeds[concept]["opinions"]:
            data[concept]["opinions"].append(opinion_expendor.expand(o, aspects=seeds[concept]["aspects"]))
    
    print("\n" * 3)


    # Step 2: Randomly sample terms to generate the dataset
    print("#########################################")
    print("<<<  Randomly generating the dataset  >>>")
    print("#########################################")

    num_positive_pairs = 0
    num_negative_pairs = 0

    with open(output_filename, 'w') as f:
        for _ in tqdm.tqdm(range(size)):
            roulette = random.random() # If roulette == 1 ==> Force a positive example
                                        # If roulette == 0 ==> Randomly take either a pos or neg example
            if roulette < min_ratio_pos:
                # Sample a concept
                c = random.choice(concepts)

                # Sample two random aspects from this concept
                a1 = random.choice(data[c]["aspects"])
                a2 = random.choice(data[c]["aspects"])

                # Sample an opinion set from that concept
                op = random.choice(data[c]["opinions"])

                # Sample two opnions from that opinion set
                o1 = random.choice(op)
                o2 = random.choice(op)

                num_positive_pairs += 1
                f.write("{} {}, {} {}, 1\n".format(o1, a1, o2, a2))

            
            else:
                c1 = random.choice(concepts)
                a1 = random.choice(data[c1]["aspects"])
                o1_index = random.randint(0, len(data[c1]["opinions"]) - 1)
                o1 = random.choice(data[c1]["opinions"][o1_index])

                c2 = random.choice(concepts)
                a2 = random.choice(data[c2]["aspects"])
                o2_index = random.randint(0, len(data[c2]["opinions"]) - 1)
                o2 = random.choice(data[c2]["opinions"][o2_index])

                sim = 1 if c1 == c2 and o1_index == o2_index else 0
                if sim == 1:
                    num_positive_pairs += 1
                else:
                    num_negative_pairs += 1
                f.write("{} {}, {} {}, {}\n".format(o1, a1, o2, a2, sim))




        print("==> Dataset is generated successfully")
        print("POS ==> ", num_positive_pairs)
        print("NEG ==> ", num_negative_pairs)
        print("% POS ==>", math.floor((num_positive_pairs / size) * 100), "%")

   




if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # Essential arguments
    parser.add_argument("--seeds", type=str, required=True, help="Path to the file containing the original seeds")
    parser.add_argument("--output_filename", type=str, required=True, help="filepath to the newly generated training dataset")
    parser.add_argument("--expansion_config", type=str, required=True, help="Filepath to configuration file containing parametrizations of aspect and opinion expansion methods")
    
    # Seed expansion related arguments
    parser.add_argument("--as_min_consensus_rate", type=float, default=0.2, help="Minimum consensus rate for aspect expansion methods to add a given word into the final list of expansions")
    parser.add_argument("--op_filter_strategy", type=str, default="both", choices=["both", "lm", "mlm"], help="Filtering stratgey used to filter expansions of opinion terms")
    parser.add_argument("--op_lm", type=str, default="gpt2", help="Language Model used to filter expansions of opinion terms")
    parser.add_argument("--op_mlm", type=str, default="bert-base-uncased", help="Masked Language Model used to filter expansions of opinion terms")
    
    # Other arguments for dataset creation
    parser.add_argument("--min_ratio_pos", type=float, default=0.3, help="Minimum ratio of postive examples in the to-be generated dataset")
    parser.add_argument("--dataset_size", type=int, default=100000, help="Number of examples in the generated dataset")

    args = parser.parse_args()

    print(vars(args))



    seeds = read_seed_words(args.seeds)
    generate_new_dataset(seeds, args.expansion_config, args.output_filename, args.dataset_size, args.min_ratio_pos, args.as_min_consensus_rate)