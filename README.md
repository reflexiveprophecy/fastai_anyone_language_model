## Fast AI Anyone Language Model

This is a project that helps to efficiently streamline the process of training a language model to imitate anyone's conversational style, using the example of several famous people's tweets. 

Warning: We are not responsible for any copyright, ethical or legal issues associated with the project. Use this project's code at your own risks. 

After git clone the repository, please run the following commands to be able to use the project:

    python3 -m venv venv
    pip install -r requirements.txt

Please note that after .lr_find() is run, a learning rate graph will be generated and named as learning_rate_graph.png in the data directory, you would need to manually input the range of desired learning rates based on where losses descend the fastest on the chart, for the following training process. 


