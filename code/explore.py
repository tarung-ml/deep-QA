import matplotlib.pyplot as plt
plt.interactive(False) # maybe pycharm specific

def lengthHistogram(filename):
    lengths = []
    with open('data/squad/' + filename) as f:
        for line in f:
            lengths.append(len(line.split(" ")))

    plt.hist(lengths)
    plt.title("Histogram of text-length, max_size = %s" % max(lengths))
    plt.xlabel(filename + " length")
    plt.savefig("plots/" + filename +"_histogram.pdf", bbox_inches="tight"); plt.clf(); plt.close()

if __name__ == "__main__":
    filenames = ['train.ids.question', 'train.ids.context', 'train.answer']
    for filename in filenames:
        lengthHistogram(filename)
# question ~ 10-20 words
# context ~ 100-200 words
# answer ~ 0-10 wordss
