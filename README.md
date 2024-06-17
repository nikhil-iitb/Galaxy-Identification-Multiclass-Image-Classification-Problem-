How to run the program?
<br>
One way is to use 
<br>
`python3 train.py all <model_name> <epochs> <lr>`
<br>
model_name can be 'inception' or 'cnn'
<br><br>
Possible error "from paths import ..." paths is not recognised => change to "from .paths import ...." it works
<br><br>
Best way to check each functionality is to use terminal and run separate function as follows:
<br><br>

Please go through data.py which is used to prepare data, the file is well commented to understand and pre-process data
<br>
Import functions in terminal and run them to generate results
<br><br>
Please uncomment create_and_save_model_input() in data.py