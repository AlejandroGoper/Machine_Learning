#from matplotlib.animation import animation
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation

"""
perceptron:
   - a supervised learning model which determines (via the perceptron learning algorithm) whether or not an input 
     defined by its associated feature vector belongs to a particular class
   - this model assumes the training_data is linearly separable (i.e. a hyperplane can completely and correctly
     divide the training_data)
   - this model also assumes output is binary in the form of +1 or -1

"""
class perceptron:
    """
    __init__: constructor
    """
    def __init__(self, training_data, max_iterations, num_features, max_x):
        self.w = np.zeros((num_features + 1, 1))    # index 0 is bias weight; indices 1 and 2 are the 2 dimensions of input vectors x_i
        self.training_data = training_data
        self.max_iterations = max_iterations
        self.iteration_text = ""
        self.x = []
        self.line = []
        self.max_x = max_x

    """
    __sign__: returns +1 if number is greater than 0; otherwise, -1 
    """
    def __sign__(self, num):
        return 1 if num > 0. else -1

    """
    train: train perceptron using training data; return the weighting vector which perfectly classifies the training_data
    """
    def train(self):
        t = 0
        while t < self.max_iterations: 
            data_val = self.training_data[np.random.randint(0, len(self.training_data))]
            if self.__sign__(np.dot(np.transpose(self.w), data_val[0])) != data_val[1]:   # Misclassified Item
                self.w = self.w + (data_val[1] * data_val[0])
            t += 1
        return self.w

    """
    __weight_line__: return y values for the line
    """
    def __weight_line__(self, x):
        y = np.linspace(-self.max_x[0], self.max_x[0], len(x))
        if self.w[2][0] != 0:
            line_slope = ((-1. * self.w[1][0]) - self.w[0][0]) / self.w[2][0]
            for index, val in enumerate(y):
                y[index] = line_slope * x[index]
        return y

    """
    __animate__: update the line by following the perceptron learning algorithm; update iteration_text too
    """
    def __animate__(self, i):
        data_val = self.training_data[np.random.randint(0, len(self.training_data))]
        if self.__sign__(np.dot(np.transpose(self.w), data_val[0])) != data_val[1]:   # Misclassified Item
            self.w = self.w + (data_val[1] * data_val[0])
        self.line.set_ydata(self.__weight_line__(self.x))
        self.iteration_text.set_text('n = ' + str(len(self.training_data)) + '; t = ' + str(i))
        return self.line, self.iteration_text

    """
    __init_anim__: since blit=True, this is required to initialize the animation
    """
    def __init_anim__(self):
        self.line.set_ydata(np.ma.array(self.x, mask=True))
        self.iteration_text.set_text('')
        return self.line, self.iteration_text

    """
    __assign_sym_col__: internal method to assign different colors to pass/fail (+1, -1) points
    """
    def __assign_sym_col__(self):
        symbols = []; colors = []
        for val in self.training_data:
            sym = 'o'; col = 'red' if val[1] == 1 else 'yellow'
            symbols.append(sym); colors.append(col)
        return (symbols, colors)

    """
    training_animation: run the perceptron and produce an animation to go along with it
    """
    def training_animation(self):
        fig, ax = plt.subplots()

        # Add training data points; label pass/fail (+1,-1) points differently
        (symbols, colors) = self.__assign_sym_col__()
        for _s, c, _x, _y in zip(symbols, colors, self.training_data.get_x_1(), self.training_data.get_x_2()):
            ax.scatter(_x, _y, s=100, marker=_s, c=c)

        # Set plot's properties: xlim, ylim, title, iteration_text
        ax.set_xlim([-self.max_x[1], self.max_x[1]]); ax.set_ylim([-self.max_x[2], self.max_x[2]])
        plt.title('Perceptron Learning Algorithm Animation', loc='left')
        self.iteration_text = ax.text(0.81, 1.015, '', transform=ax.transAxes)

        # Set up line to be animated
        self.x = np.arange(-max_x[1], max_x[1], 0.01)
        self.line, = ax.plot(self.x, np.zeros(len(self.x)))
        plt.setp(self.line, linewidth=2.0)

        # Animate
        anim = animation.FuncAnimation(fig, self.__animate__, np.arange(1, self.max_iterations), init_func=self.__init_anim__,
                                      interval=15, blit=True)

        # Save figure; requires an animation-writing scheme is setup such as ffmpeg or mencoder
        # IF YOU DO NOT HAVE THESE INSTALLED, COMMENT OUT THE LINE BELOW
        anim.save('perceptron_animation.gif', fps=30)
        # AND UNCOMMENT THE LINE BELOW
        # plt.show()
        

"""
training_set: class wrapper for training data; produces a linearly separable data set
"""
class training_set:
    """
    __init__: constructor
              num_elements is the size of the training set; num_features is the dimension of the input data
              max_x[1] is the magnitude of maximum value for input vectors in element 1; likewise for max_x[2]
              recall x[0] is always 1.0 for every input vector x
    """
    def __init__(self, num_elements, max_x):
        self.num_elements = num_elements
        self.data_set = []   # list of tuples of the form (x, y), where x is a 2-dimensional vector
        self.index = 0
        self.__create_data_set__(max_x)

    """
    __create_data_set__: create randomized data with certain limitations
    """
    def __create_data_set__(self, max_x):
        for index in range(self.num_elements):
            x_1 = np.random.uniform(-max_x[1], max_x[1]); x_2 = np.random.uniform(-max_x[2], max_x[2])
            x = np.array([[1.0], [x_1], [x_2]])  # x[0] is fixed to be 1.0
            y = -1 if x_2 < 0. else 1    # assign -1 to values below x_0 axis & +1 to values above x_0 axis
            self.data_set.append((x,y))

    """
    get_x_1: returns an array of each input vector x's index 1 
    """
    def get_x_1(self):
        x_1 = np.arange(len(self.data_set))
        for index in np.arange(len(self.data_set)):
            x_1[index] = self.data_set[index][0][1]
        return x_1

    """
    get_x_2: returns an array of each input vector x's index 2
    """
    def get_x_2(self):
        x_2 = np.arange(len(self.data_set))
        for index in np.arange(len(self.data_set)):
            x_2[index] = self.data_set[index][0][2]
        return x_2
    
    """
    __iter__: overrides __iter__
    """
    def __iter__(self):
        return self

    """
    __next__: overrides __next__
    """
    def __next__(self):
        try:
            data_val = self.data_set[self.index]
        except IndexError:
            raise StopIteration
        self.index += 1
        return data_val

    """
    __len__: overrides __len__
    """
    def __len__(self):
        return self.num_elements

    """
    __getitem__: overrides __getitem__
    """
    def __getitem__(self, key):
        return self.data_set[key]
    
    """
    get_data: basic accessor to return list
    """
    def get_data(self):
        return self.data_set


"""
test driver: only executed when run as "python perceptron.py"; not as import
"""
if __name__ == "__main__":
    max_iterations = 100; num_elements = 250; num_features = 2; max_x = (1, 25, 25)
    training_dataset = training_set(num_elements, max_x)
    model = perceptron(training_dataset, max_iterations, num_features, max_x)
    model.training_animation()
