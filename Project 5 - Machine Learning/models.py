import nn
import time

class PerceptronModel(object):
    def __init__(self, dimensions):
        """
        Initialize a new Perceptron instance.

        A perceptron classifies data points as either belonging to a particular
        class (+1) or not (-1). `dimensions` is the dimensionality of the data.
        For example, dimensions=2 would mean that the perceptron must classify
        2D points.
        """
        self.w = nn.Parameter(1, dimensions)

    def get_weights(self):
        """
        Return a Parameter instance with the current weights of the perceptron.
        """
        return self.w

    def run(self, x):
        """
        Calculates the score assigned by the perceptron to a data point x.

        Inputs:
            x: a node with shape (1 x dimensions)
        Returns: a node containing a single number (the score)
        """
        "*** YOUR CODE HERE ***"
        return nn.DotProduct(x, self.w)

    def get_prediction(self, x):
        """
        Calculates the predicted class for a single data point `x`.

        Returns: 1 or -1
        """
        "*** YOUR CODE HERE ***"
        if nn.as_scalar(self.run(x)) >= 0:
            return 1
        else:
            return -1

    def train(self, dataset):
        """
        Train the perceptron until convergence.
        """
        "*** YOUR CODE HERE ***"
        mistakeMade = True
        while mistakeMade:
            mistakeMade = False
            for x, y in dataset.iterate_once(1):
                y = nn.as_scalar(y)
                if self.get_prediction(x) != y:
                    mistakeMade = True
                    self.w.update(x, y)


class RegressionModel(object):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    """
    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        self.layer1_size = 100
        self.layer2_size = 100 # Set to 1 to bascially remove second hidden layer
        self.w01 = nn.Parameter(1, self.layer1_size)
        self.b1 = nn.Parameter(1, self.layer1_size)
        # self.w12 = nn.Parameter(self.layer1_size, 1)
        # self.b2 = nn.Parameter(1, 1)
        self.w12 = nn.Parameter(self.layer1_size, self.layer2_size)
        self.b2 = nn.Parameter(1, self.layer2_size)
        self.w23 = nn.Parameter(self.layer2_size, 1)
        self.b3 = nn.Parameter(1, 1)
        self.learning_rate = 0.1

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
        Returns:
            A node with shape (batch_size x 1) containing predicted y-values
        """
        "*** YOUR CODE HERE ***"
        layer1_values = nn.ReLU(nn.AddBias(nn.Linear(x, self.w01), self.b1))
        layer2_values = nn.ReLU(nn.AddBias(nn.Linear(layer1_values, self.w12), self.b2))
        predicted_y = nn.AddBias(nn.Linear(layer2_values, self.w23), self.b3)

        return predicted_y

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
            y: a node with shape (batch_size x 1), containing the true y-values
                to be used for training
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        predicted_y = self.run(x)
        loss = nn.SquareLoss(predicted_y, y)

        return loss

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        loss_threshold = 0.02 / 10
        batch_size = 200

        for x, y in dataset.iterate_forever(batch_size):
            loss = self.get_loss(x, y)
            # print(nn.as_scalar(loss))
            if nn.as_scalar(loss) < loss_threshold:
                break

            parameters = [self.w01, self.b1, self.w12, self.b2, self.w23, self.b3]

            gradients = nn.gradients(loss, parameters)

            for i in range(len(parameters)):
                parameters[i].update(gradients[i], -self.learning_rate)


class DigitClassificationModel(object):
    """
    A model for handwritten digit classification using the MNIST dataset.

    Each handwritten digit is a 28x28 pixel grayscale image, which is flattened
    into a 784-dimensional vector for the purposes of this model. Each entry in
    the vector is a floating point number between 0 and 1.

    The goal is to sort each digit into one of 10 classes (number 0 through 9).

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        self.input_layer_size = 28 * 28
        self.layer1_size = 800
        # self.layer2_size = 64
        # self.layer3_size = 64
        self.output_layer_size = 10
        self.w01 = nn.Parameter(self.input_layer_size, self.layer1_size)
        self.b1 = nn.Parameter(1, self.layer1_size)
        self.w12 = nn.Parameter(self.layer1_size, self.output_layer_size)
        self.b2 = nn.Parameter(1, self.output_layer_size)
        # self.w12 = nn.Parameter(self.layer1_size, self.layer2_size)
        # self.b2 = nn.Parameter(1, self.layer2_size)
        # self.w23 = nn.Parameter(self.layer2_size, self.output_layer_size)
        # self.b3 = nn.Parameter(1, self.output_layer_size)
        # self.w23 = nn.Parameter(self.layer2_size, self.layer3_size)
        # self.b3 = nn.Parameter(1, self.layer3_size)
        # self.w34 = nn.Parameter(self.layer3_size, self.output_layer_size)
        # self.b4 = nn.Parameter(1, self.output_layer_size)
        self.learning_rate = 1

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Your model should predict a node with shape (batch_size x 10),
        containing scores. Higher scores correspond to greater probability of
        the image belonging to a particular class.

        Inputs:
            x: a node with shape (batch_size x 784)
        Output:
            A node with shape (batch_size x 10) containing predicted scores
                (also called logits)
        """
        "*** YOUR CODE HERE ***"
        layer1_values = nn.ReLU(nn.AddBias(nn.Linear(x, self.w01), self.b1))
        predicted_y = nn.AddBias(nn.Linear(layer1_values, self.w12), self.b2)
        # layer2_values = nn.ReLU(nn.AddBias(nn.Linear(layer1_values, self.w12), self.b2))
        # predicted_y = nn.AddBias(nn.Linear(layer2_values, self.w23), self.b3)
        # layer3_values = nn.ReLU(nn.AddBias(nn.Linear(layer2_values, self.w23), self.b3))
        # predicted_y = nn.AddBias(nn.Linear(layer3_values, self.w34), self.b4)

        return predicted_y

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 10). Each row is a one-hot vector encoding the correct
        digit class (0-9).

        Inputs:
            x: a node with shape (batch_size x 784)
            y: a node with shape (batch_size x 10)
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        predicted_y = self.run(x)
        loss = nn.SoftmaxLoss(predicted_y, y)

        return loss

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        accuracy_threshold = 0.98
        batch_size = 100
        last_test = time.time()

        for x, y in dataset.iterate_forever(batch_size):
            if time.time() - last_test > 5:
                last_test = time.time()
                if dataset.get_validation_accuracy() >= accuracy_threshold:
                    break

            loss = self.get_loss(x, y)
            parameters = [self.w01, self.b1, self.w12, self.b2] #, self.w23, self.b3] #, self.w34, self.b4]
            gradients = nn.gradients(loss, parameters)

            for i in range(len(parameters)):
                parameters[i].update(gradients[i], -self.learning_rate)

class LanguageIDModel(object):
    """
    A model for language identification at a single-word granularity.

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Our dataset contains words from five different languages, and the
        # combined alphabets of the five languages contain a total of 47 unique
        # characters.
        # You can refer to self.num_chars or len(self.languages) in your code
        self.num_chars = 47
        self.languages = ["English", "Spanish", "Finnish", "Dutch", "Polish"]

        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        self.num_languages = len(self.languages)
        self.hidden_size = 100
        self.w_letter = nn.Parameter(self.num_chars, self.hidden_size)
        self.w_hidden = nn.Parameter(self.hidden_size, self.hidden_size)
        self.b_hidden = nn.Parameter(1, self.hidden_size)
        self.w_final = nn.Parameter(self.hidden_size, self.num_languages)
        self.b_final = nn.Parameter(1, self. num_languages)

        self.learning_rate = 0.1

    def run(self, xs):
        """
        Runs the model for a batch of examples.

        Although words have different lengths, our data processing guarantees
        that within a single batch, all words will be of the same length (L).

        Here `xs` will be a list of length L. Each element of `xs` will be a
        node with shape (batch_size x self.num_chars), where every row in the
        array is a one-hot vector encoding of a character. For example, if we
        have a batch of 8 three-letter words where the last word is "cat", then
        xs[1] will be a node that contains a 1 at position (7, 0). Here the
        index 7 reflects the fact that "cat" is the last word in the batch, and
        the index 0 reflects the fact that the letter "a" is the inital (0th)
        letter of our combined alphabet for this task.

        Your model should use a Recurrent Neural Network to summarize the list
        `xs` into a single node of shape (batch_size x hidden_size), for your
        choice of hidden_size. It should then calculate a node of shape
        (batch_size x 5) containing scores, where higher scores correspond to
        greater probability of the word originating from a particular language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
        Returns:
            A node with shape (batch_size x 5) containing predicted scores
                (also called logits)
        """
        "*** YOUR CODE HERE ***"
        f = nn.ReLU(nn.AddBias(nn.Linear(xs[0], self.w_letter), self.b_hidden))
        for letter_i in range(1, len(xs)):
            f = nn.ReLU(nn.AddBias(nn.Add(nn.Linear(xs[letter_i], self.w_letter), nn.Linear(f, self.w_hidden)), self.b_hidden))

        predicted_y = nn.AddBias(nn.Linear(f, self.w_final), self.b_final)

        return predicted_y

    def get_loss(self, xs, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 5). Each row is a one-hot vector encoding the correct
        language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
            y: a node with shape (batch_size x 5)
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        predicted_y = self.run(xs)
        loss = nn.SoftmaxLoss(predicted_y, y)

        return loss

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        accuracy_threshold = 0.85
        batch_size = 100
        last_test = time.time()

        for x, y in dataset.iterate_forever(batch_size):
            if time.time() - last_test > 3:
                last_test = time.time()
                if dataset.get_validation_accuracy() >= accuracy_threshold:
                    break

            loss = self.get_loss(x, y)
            parameters = [self.w_letter, self.w_hidden, self.b_hidden, self.w_final, self.b_final]
            gradients = nn.gradients(loss, parameters)

            for i in range(len(parameters)):
                parameters[i].update(gradients[i], -self.learning_rate)
