#python 3.7

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
    #find the best learning rate
    learner.lr_find()
    #plot the learning rate graph vs. loss
    fig = learner.recorder.plot(return_fig = True, skip_end = 15)
    #saving the learning rate chart as learning_rate_graph.png
    fig.savefig(learning_rate_fig_path + 'learning_rate_graph.png')
    return None


def model_training(learner, export_file = None, num_of_epochs = 3, moms=(0.8,0.7)):
    '''this function trains the model and export the learner and weights'''
    #input the learning rate range based on the outputted learning_rate_graph
    refined_learning_rate = input('please input the selected learning rate min and max, separated by comma, i.e. 3e-5,3e-4: ')
    learning_rate_min, learning_rate_max = [float(x) for x in refined_learning_rate.split(',')]
    #train the model with fastai's fit_one_cycle
    learner.fit_one_cycle(num_of_epochs, max_lr=slice(learning_rate_min, learning_rate_max), moms = moms)
    #export everything with learner and model weights, do not use learner.save() as we are also trying to save the learner as well
    learner.export(file = export_file)
    return learner


def main(dataset = None):
    '''this function wraps everything together'''
    prepared_data = data_preparation(data_path, dataset, cols = 'Tweet')
    #use the AWD_LSTM: Wikitext 103 pre-trained model for transfer learning
    language_learner = lm_learner(prepared_data, model = AWD_LSTM)
    tune_learning_rate(language_learner)
    trained_model = model_training(language_learner, export_file = model_path + 'trained_{}_lm.pkl'.format(re.sub('data_|.csv', '', dataset)))
    #print the model summary 
    print(trained_model.summary())
   

if __name__ == '__main__':
    main(dataset = 'data_trump.csv')



