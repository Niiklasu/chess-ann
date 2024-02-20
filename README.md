# Evaluate chess positions using an Artificial Neural Network
## Introduction 
This project was created as part of my university class. A deep neural network is trained on six million chess games, equivalent to 480 million unique positions, and evaluated by Stockfish to learn how to evaluate any chess position. The trained model is compared on a held-out test set against simpler evaluation functions. As proof of concept, the model is used as the evaluation function for a Minimax algorithm to create a simple bot to play chess against. 

Some preprocessing is necessary, which is explained in [this file](preprocessing/how.txt). A deeper analysis can be found in the appended [conference paper](conference_paper.pdf).
