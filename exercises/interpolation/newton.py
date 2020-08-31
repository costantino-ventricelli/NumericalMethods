# Python3 program for implementing 
# Newton divided difference formula

from exercises.interpolation.chebyshev_polynomial import ChebyshevNodes
import numpy as np
import matplotlib.pyplot as plot

def f(x):
    return 1 / (1 + 25 * x ** 2)

# Function to find the product term 
def proterm(i, value, x):
    pro = 1
    for j in range(i):
        pro = pro * (value - x[j])
    return pro


# Function for calculating 
# divided difference table 
def dividedDiffTable(x, y, n):
    for i in range(1, n):
        for j in range(n - i):
            y[j][i] = ((y[j][i - 1] - y[j + 1][i - 1]) /
                       (x[j] - x[i + j]))
    return y


# Function for applying Newton's 
# divided difference formula 
def applyFormula(value, x, y, n):
    sum = y[0][0]

    for i in range(1, n):
        sum = sum + (proterm(i, value, x) * y[0][i])

    return sum


# Function for displaying divided 
# difference table 
def printDiffTable(y, n):
    for i in range(n):
        for j in range(n - i):
            print(round(y[i][j], 4), "\t",
                  end=" ")

        print("")

    # Driver Code 


# number of inputs given 
n = 20
y = [[0 for i in range(n)]
     for j in range(n)]
x = ChebyshevNodes.get_chebyshev_nodes(-1, 1, 20)

# y[][] is used for divided difference 
# table where y[][0] is used for input 
for i in range(len(x)):
    y[i][0] = f(x[i])

# calculating divided difference table 
y = dividedDiffTable(x, y, n)

# displaying divided difference table 
printDiffTable(y, n)

x_value = np.linspace(x[0], x[n - 1], 300)
polynomial = np.zeros(300)

for i in range(len(x_value)):
    polynomial[i] = applyFormula(x_value[i], x, y, n)

plot.close('all')
plot.figure(0)
plot.plot(x_value, polynomial, label="p(x)")
plot.plot(x_value, f(x_value), label="f(x)")
plot.legend()
plot.show()

# This code is contributed by mits 
