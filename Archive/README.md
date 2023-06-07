Directory Structure:

- README.md
- code/
  - **dev.py**
  - (supporting .py files)
  - data/
    - (testing and training data)
- results/
  - (save folder for map plots)

Tips:

Run the dev.py module. You can change the output and running conditions in the hyper parameter dictionary at the top of the code. For example, you can turn on live plotting, change the map scale, etc.

dev.py will only run on the test data. You can change this within the run function input. ([4,5] will run the test data; [1,2,3] will run the training)