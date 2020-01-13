from fastai.text import load_learner

model_path = './models/'

def model_prediction(model_file = None, num_of_words = 30):
    '''this function loads the model and predict with a sentence'''
    model = load_learner(model_path, file = model_file)
    # model = learner.load('trained_model')
    print(model.predict('Donald Trump', num_of_words, temperature = 0.75))
    print(model.predict('Women', num_of_words, temperature = 0.75))
    print(model.predict('Democrates', num_of_words, temperature = 0.75))


if __name__ == '__main__':
    model_prediction(model_file = 'trained_hillary_lm.pkl', num_of_words = 40)





