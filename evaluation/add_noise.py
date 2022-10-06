import random, argparse

def read_training_data(filename):
    phrases = []
    labels = []
    with open(filename, 'r') as f:
        for l in f.readlines():
            s1, s2, sim = l.split(',')
            phrases.append((s1.strip(' '), s2.strip(' ')))
            labels.append(float(sim.strip('\n ')))
    return phrases, labels



def write_training_data(filename, phrases, labels):
    assert len(phrases) == len(labels)
    with open(filename, 'w') as f:
        for i in range(len(phrases)):
            f.write("{}, {}, {}\n".format(phrases[i][0], phrases[i][1], labels[i]))



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    
    parser.add_argument("--dataset", type=str, required=True, help="Path to dataset into which to introduce noise")
    parser.add_argument("--output", type=str, required=True, help="Path to the perturbed output dataset")
    parser.add_argument("--noise_ratio", type=float, default=0.1, help="Ratio of dataset to perturb")
    args = parser.parse_args()


    phrases, labels = read_training_data(args.dataset)
    assert len(phrases) == len(labels)
    num_training_example = len(phrases)
    print('Training dataset read')

    new_labels = []

    for l in labels:
        likelihood = random.random()
        new_label = int(l)
        if likelihood < args.noise_ratio:
            new_label = 1 if l == 0 else 0
        new_labels.append(new_label)

    write_training_data(args.output, phrases, new_labels)
        
