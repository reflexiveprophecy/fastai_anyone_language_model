import pandas as pd
import fastai
import re
from fastai.text import TextList, language_model_learner, load_data, fit_one_cycle, AWD_LSTM, load_learner

data_path = './data_sets/'
model_path = './models/'
learning_rate_fig_path = './learning_rate/'


def data_preparation(file_path, dataset, cols = None, bs = 48):
    '''this function prepares the data by using TextList from FastAI datablock API'''
    data_df = pd.read_csv(file_path + dataset, encoding = 'latin-1')
    data_lm = (TextList.from_df(data_df, cols = cols) \
           #Inputs: from csv file we have in dataset
            .split_by_rand_pct(0.1) \
           #We randomly split and keep 10% for validation
            .label_for_lm() \
           #We want to do a language model so we label accordingly
            .databunch(bs=bs))

    return data_lm


def lm_learner(data, model = None, drop_mult = 0.3):
    '''this function gets the learner object and returns the object'''
    learner = language_model_learner(data, model, drop_mult = drop_mult)
    return learner


def tune_learning_rate(learner):
    '''this function plots the learning rate vs. loss and saves the graph'''
    learner.lr_find()
    fig = learner.recorder.plot(return_fig = True, skip_end = 15)
    #saving the learning rate chart as learning_rate_graph.png
    fig.savefig(learning_rate_fig_path + 'learning_rate_graph.png')
    return None


def model_training(learner, export_file = None, num_of_epochs = 3, moms=(0.8,0.7)):
    '''this function trains the model and export the learner and weights'''
    refined_learning_rate = input('please input the selected learning rate min and max, separated by comma, i.e. 3e-5,3e-4: ')
    learning_rate_min, learning_rate_max = [float(x) for x in refined_learning_rate.split(',')]
    learner.fit_one_cycle(num_of_epochs, max_lr=slice(learning_rate_min, learning_rate_max), moms = moms)
    # learner.save('trained_model', return_path = True)
    learner.export(file = export_file)
    return learner


def main(dataset = None):
    '''this function wraps everything together'''
    prepared_data = data_preparation(data_path, dataset, cols = 'Tweet')
    language_learner = lm_learner(prepared_data, model = AWD_LSTM)
    tune_learning_rate(language_learner)
    trained_model = model_training(language_learner, export_file = model_path + 'trained_{}_lm.pkl'.format(re.sub('data_|.csv', '', dataset)))
    print(trained_model.summary())
    # model_prediction()
    # model_prediction(language_learner)


if __name__ == '__main__':
    main(dataset = 'data_trump.csv')



