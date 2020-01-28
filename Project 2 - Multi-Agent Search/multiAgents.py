# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util
import math

from game import Agent, Actions

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        close_ghosts = sum(-200 for ghost_state in newGhostStates if util.manhattanDistance(ghost_state.getPosition(), newPos) <= 1)
        
        if successorGameState.isWin():
            return 10000

        seen = set()
        seen.add(newPos)
        queue = util.Queue()
        queue.push( (newPos, 0) )
        dist_to_food = None
        walls = successorGameState.getWalls()
        while dist_to_food == None:
            current_pos, dist = queue.pop()
            for direction in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
                x, y = current_pos
                dx, dy = Actions.directionToVector(direction)
                next_x, next_y = next_pos = (int(x + dx), int(y + dy))
                if not walls[next_x][next_y] and next_pos not in seen:
                    seen.add( next_pos )
                    queue.push( (next_pos, dist + 1) )
                    if newFood[next_x][next_y]:
                        dist_to_food = dist + 1

        return successorGameState.getScore() + close_ghosts + 10 / (dist_to_food + 1)

def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        bestScore, bestAction = self.minimaxNode(gameState, 0, 0)
        return bestAction

    def minimaxNode(self, gameState, agentIndex, depth):
        if depth >= self.depth or gameState.isWin() or gameState.isLose():
            return (self.evaluationFunction(gameState), None)

        if agentIndex == 0:
            bestScore = -math.inf
            for action in gameState.getLegalActions(agentIndex):
                score, _ = self.minimaxNode(gameState.generateSuccessor(agentIndex, action), (agentIndex + 1) % gameState.getNumAgents(), depth + (1 if agentIndex == gameState.getNumAgents() - 1 else 0))
                if score > bestScore:
                    bestScore, bestAction = score, action

        else:
            bestScore = math.inf
            for action in gameState.getLegalActions(agentIndex):
                score, _ = self.minimaxNode(gameState.generateSuccessor(agentIndex, action), (agentIndex + 1) % gameState.getNumAgents(), depth + (1 if agentIndex == gameState.getNumAgents() - 1 else 0))
                if score < bestScore:
                    bestScore, bestAction = score, action

        return (bestScore, bestAction)

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        bestScore, bestAction = self.alphaBetaNode(gameState, 0, 0, -math.inf, math.inf)
        return bestAction

    def alphaBetaNode(self, gameState, agentIndex, depth, alpha, beta):
        if depth >= self.depth or gameState.isWin() or gameState.isLose():
            return (self.evaluationFunction(gameState), None)

        if agentIndex == 0:
            bestScore = -math.inf
            for action in gameState.getLegalActions(agentIndex):
                score, _ = self.alphaBetaNode(gameState.generateSuccessor(agentIndex, action), (agentIndex + 1) % gameState.getNumAgents(), depth + (1 if agentIndex == gameState.getNumAgents() - 1 else 0), alpha, beta)
                if score > bestScore:
                    bestScore, bestAction = score, action

                if bestScore > beta:
                    return (bestScore, bestAction)
                alpha = max(alpha, bestScore)

        else:
            bestScore = math.inf
            for action in gameState.getLegalActions(agentIndex):
                score, _ = self.alphaBetaNode(gameState.generateSuccessor(agentIndex, action), (agentIndex + 1) % gameState.getNumAgents(), depth + (1 if agentIndex == gameState.getNumAgents() - 1 else 0), alpha, beta)
                if score < bestScore:
                    bestScore, bestAction = score, action
                    
                if bestScore < alpha:
                    return (bestScore, bestAction)
                beta = min(beta, bestScore)

        return (bestScore, bestAction)

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        bestScore, bestAction = self.expectimaxNode(gameState, 0, 0)
        return bestAction

    def expectimaxNode(self, gameState, agentIndex, depth):
        if depth >= self.depth or gameState.isWin() or gameState.isLose():
            return (self.evaluationFunction(gameState), None)

        if agentIndex == 0:
            bestScore = -math.inf
            for action in gameState.getLegalActions(agentIndex):
                score, _ = self.expectimaxNode(gameState.generateSuccessor(agentIndex, action), (agentIndex + 1) % gameState.getNumAgents(), depth + (1 if agentIndex == gameState.getNumAgents() - 1 else 0))
                if score > bestScore:
                    bestScore, bestAction = score, action
            return (bestScore, bestAction)

        else:
            avgScore = 0
            n_actions = 0
            for action in gameState.getLegalActions(agentIndex):
                score, _ = self.expectimaxNode(gameState.generateSuccessor(agentIndex, action), (agentIndex + 1) % gameState.getNumAgents(), depth + (1 if agentIndex == gameState.getNumAgents() - 1 else 0))
                avgScore += score
                n_actions += 1
            avgScore /= n_actions
            return (avgScore, None)
        

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
