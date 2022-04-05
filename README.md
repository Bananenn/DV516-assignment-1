
# DV516-assignment-1

Assignment 1 <br>
By: André Franzén (af223kr) <br>
Course: 1DV516 <br>
Date: March-April 2022


# Task 1
File name `microships.py`
### How to run
In order to run task 1.2 Where the predefined points are calculated run the `perdictForPredefPoints()` (In order to avoid messy output) <br>
To run Task 1.3 and display the plots just run the python file

### Comments
I have gone through many many iterations to try to make it quicker but and have to great lengths suceeded. I have cut down the time it takes from my original working solution by more than 75% But it still takes some time to run and I Think its due to how i calculate the distances

```python
for x,y,ok in points:
        #d=√((x_2-x_1)²+(y_2-y_1)²)
        dis = sqrt( pow((z[0] - x),2) + pow((z[1] - y),2) )
        distanceList.append( ((x, y, ok),dis) )
```
I want to great lengths to try to understand and how to use distance matricas and there for be able to use brodcast but i did not suceed.

### Result 
![img not found](Pictures/task1.png)
Time to complete (on my pc) 6.5s to compute above image

<br>

# Task 2
File name `polynomial.py`
### How to run
Run the python file, (2 plots will open 1 corresponding to task 1.2) and the other with the MSE errors and regression lines

### Comments
For this task I am happy about my implementation there are probably better ways as i use a fair few loops but all in all its pretty quick.

Note! the MSE should in theroy be 0 for the one with K = 1 but due to the dots being to close together even a very very small stepsize is needed.
<br>
2.5: Which K gives the best regression Motivate your answer! I Think that the K = 9 is the best K this is due to it having the lowest MSE test error. training error does not really give alot of insight this is why i make my pick based on MSE test value

### Result 
![img not found](Pictures/Task2.png)
Time to complete (on my pc) 6s to compute above image

# Task 3 - VG
File name ``MNIST_V2.py``
### How to run
bla bla
### Comments
bla bla
### Results

# Task 4
File name ``sckitMicroShips.py``
### How to run
Run the python file ``sckitMicroShips.py``
### Comments
I am happy with my implementation was fun to see how much easier it was with the library
### Results
![img not found](Pictures/Task4.png)
Time to complete (on my pc) 3s to compute above image