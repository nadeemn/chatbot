Project information.
The project implements a simple AI-Based chatbot designed to provide support for e-commerce platform. The chatbot can handle common queries like greetings, farewells, payment methods, return policy, promotions, order status etc. It is built using a simple fully connected feed-forward Neural Network with 2 hidden layers enabling to understand and classify user inputs to predefined classes. The chatbot is deployed using Flask and can be tested with a simple front-end design.

Approach
Data Preparation – The chat’s knowledge base is stored in a json file (‘data.json), which contains different classes with its expected inputs and possible response outputs.
Tokenization and stemming – User inputs are tokenized and then stemmed to standardize the data.
Bag of Words model – This model approach will convert the sentence into numerical numbers, representing each sentence as a numeric vector where the presence and absence of a word is changed to 1 or 0 using the predefined vocabulary set from training inputs.

Model
We used a simple feedforward Neural Network with 2 hidden layers and RELU activation functions. This will enable to learn non-linearity from the data.

Training
The data is wrapped in a custom class and loaded in batches using DataLoader  facilitated to efficiently train the data by pytorch. CrossEntropyLoss is used to measure the accuracy, and Adam optimizer updates the model weights.
Finally the model is saved to a file. This will allow to load the model late for inference without retraining.

Technologies Used
Pytorch, NLTK for basic NLP operations, JavaScript, Python, Flask, HTML, CSS.

Steps to run the project
1.	Navigate to CHATBOT-PROTOTYPE folder from your current working directory. cd CHATBOT-PROTOTYPE.
2.	Create a new virtual environment and activate it. python -m venv venv
3.	Activate the virtual env. venv/Scripts/Activate for windows .venv/bin/activate for IOS
4.	Install all the dependencies in the requirement file. pip install -r requirement.txt
5.	Run python train.py to train the model.
6.	Run flask run to start testing the chatbot. This will start a server in localhost. 

TEST
You can start with a simple greeting and then ask about different payment methods, return policies, order details, discount options, product selling etc. For the user inputs which it won't understand the bot will respond to ask again to the user, and finally, you can end the conversation by sending a farewell input.
