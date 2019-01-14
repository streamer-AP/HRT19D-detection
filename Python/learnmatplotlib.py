import matplotlib.pyplot as plt

def mpl_squares(numbers):
    squares=[value**2 for value in numbers]
    plt.plot(squares)
    plt.show()

def mpl_squares_modify(numbers):
    squares=[value**2 for value in numbers]
    plt.title("Squares",size=24)
    plt.xlabel("Values",size=14)
    plt.ylabel("Squares of Values",size=14)
    plt.tick_params(axis="both",size=14)
    plt.plot(squares)
    plt.show()

mpl_squares_modify(range(15))