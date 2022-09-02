# **Neural Network Language Model**

[![forthebadge](https://forthebadge.com/images/badges/made-with-python.svg)](https://forthebadge.com)
[![forthebadge](https://forthebadge.com/images/badges/fixed-bugs.svg)](https://forthebadge.com)

-----
## ***Objectives:***

This repository contains the codes for:
1. Neural Language Models
2. RNN-based Language Model
3. LSTM-based Language Model
4. GRU-based Language Model

The use of various statistical and probabilistic techniques to determine the likelihood of a given sequence of words appearing in a phrase is known as language modelling (LM). As a result, it is possible to describe it as a probability distribution over the vocabulary words.

-----

## ***File Structure:***

1. `LModel1.py`, `LModel3.py`, `LSTM.py`, `GRU.py` contain the Python codes for the language models. The name says it all.
2. `RollNo-ModelName-train/testpperplexity.txt` contains the perplexities scores for each sentence and the average perplexity of all the sentences in that file.
3. `Report.pdf` contains visualizations and analysis of the results of our model.
4. `Model Path.pdf` redirects you to an Outlook One Drive link containing ZIPs of all the models. The models are those keys which gave the best value (least loss) on the validation data. 
5. `extra` folder contains dump of `images` and `ipynb` versions for the code above.
-----

## ***Execution:***
The code can be executed by:
```c++
python3 <filename>.py
```
Ofcourse, you can only run those files ending with `.py`. 

When the model is run, the code snippet:
```c++
torch.save(model.state_dict(), 'model.pt')
```
stores the best parameters of the model which gives the lowest validation loss.
The code by itself calls the model back for testing purposes using:
```c++
model.load_state_dict(torch.load('model.pt'))
```
If the model has been successfully loaded, it returns
`<All keys matched successfully>`

-----
