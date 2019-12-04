import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit




def load_dataset(filename):
    """ Download the date: list of texts with scores."""
    headers = ['polarity', 'text']
    sentences = pd.read_csv(filename, encoding="utf-8", sep='\t', names=headers)
    # print distributions by rating or class
    # print(sentences.groupby('polarity').nunique())
    # return the list of rows : row = label and text
    return sentences



def save_datarows(datarows, filename):
    with open(filename, 'w', encoding= 'UTF-8', newline='\n') as f:
        for d in datarows:
            print("%s\t%s" % (d.polarity, d.text))
            f.write("%s\t%s\n" % (d.polarity, d.text))


def stratified_split(datafile):
    df = load_dataset(datafile)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
    for train_index, test_index in sss.split(df, df['polarity']):
        print("len(TRAIN):", len(train_index), "len(TEST):", len(test_index))
        print("TRAIN:", train_index, "TEST:", test_index)
        train_data = [df.loc[ind] for ind in train_index]
        test_data = [df.loc[ind] for ind in test_index]
        save_datarows(train_data, datafile+".train")
        save_datarows(test_data, datafile+".test")


if __name__ == "__main__":
    # Just for testing
    data_filename = "../data/frdataset1_train.csv"
    # stratified_split(data_filename)
    load_dataset(data_filename)
