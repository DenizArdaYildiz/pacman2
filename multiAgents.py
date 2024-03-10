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

from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.
      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gamestate):
        """
        You do not need to change this method, but you're welcome to.
        getAction chooses among the best options according to the evaluation function.
        Just like in the previous project, getAction takes a gamestate and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gamestate.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gamestate, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentgamestate, action):
        """
        Design a better evaluation function here.
        The evaluation function takes in the current and proposed successor
        gamestates (pacman.py) and returns a number, where higher numbers are better.
        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.
        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a gamestate (pacman.py)
        successorgamestate = currentgamestate.generatePacmanSuccessor(action)
        newPos = successorgamestate.getPacmanPosition()
        newFood = successorgamestate.getFood()
        newGhostStates = successorgamestate.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        # focusing on eating food.When ghost near don't go,
        newFood = successorgamestate.getFood().asList()
        minFoodist = float("inf")
        for food in newFood:
            minFoodist = min(minFoodist, manhattanDistance(newPos, food))

        # avoid ghost if too close
        for ghost in successorgamestate.getGhostPositions():
            if (manhattanDistance(newPos, ghost) < 2):
                return -float('inf')
        # reciprocal
        return successorgamestate.getScore() + 1.0/minFoodist

def scoreEvaluationFunction(currentgamestate):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.
      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentgamestate.getScore()

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

    def getAction(self, gamestate):
        """
          Returns the minimax action from the current gamestate using self.depth
          and self.evaluationFunction.
          Here are some method calls that might be useful when implementing minimax.
          gamestate.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1
          gamestate.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action
          gamestate.getNumAgents():
            Returns the total number of agents in the game
        """
        "*** YOUR CODE HERE ***"
        return self.maxval(gamestate, 0, 0)[0]

    def minimax(self, gamestate, agentIndex, depth):
        if depth is self.depth * gamestate.getNumAgents() \
                or gamestate.isLose() or gamestate.isWin():
            return self.evaluationFunction(gamestate)
        if agentIndex == 0:
            return self.maxval(gamestate, agentIndex, depth)[1]
        else:
            return self.minval(gamestate, agentIndex, depth)[1]

    def maxval(self, gamestate, agentIndex, depth):
        bestmove = ("max",-float("inf"))
        for action in gamestate.getLegalActions(agentIndex):
            succAction = (action,self.minimax(gamestate.generateSuccessor(agentIndex,action),
                                      (depth + 1)%gamestate.getNumAgents(),depth+1))
            bestmove = max(bestmove,succAction,key=lambda x:x[1])
        return bestmove

    def minval(self, gamestate, agentIndex, depth):
        bestmove = ("min",float("inf"))
        for action in gamestate.getLegalActions(agentIndex):
            succAction = (action,self.minimax(gamestate.generateSuccessor(agentIndex,action),
                                      (depth + 1)%gamestate.getNumAgents(),depth+1))
            bestmove = min(bestmove,succAction,key=lambda x:x[1])
        return bestmove


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gamestate):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        return self.maxval(gamestate, 0, 0, -float("inf"), float("inf"))[0]

    def alphabeta(self, gamestate, agentIndex, depth, alpha, beta):
        if depth is self.depth * gamestate.getNumAgents() \
                or gamestate.isLose() or gamestate.isWin():
            return self.evaluationFunction(gamestate)
        if agentIndex == 0:
            return self.maxval(gamestate, agentIndex, depth, alpha, beta)[1]
        else:
            return self.minval(gamestate, agentIndex, depth, alpha, beta)[1]

    def maxval(self, gamestate, agentIndex, depth, alpha, beta):
        bestmove = ("max",-float("inf"))
        for action in gamestate.getLegalActions(agentIndex):
            succAction = (action,self.alphabeta(gamestate.generateSuccessor(agentIndex,action),
                                      (depth + 1)%gamestate.getNumAgents(),depth+1, alpha, beta))
            bestmove = max(bestmove,succAction,key=lambda x:x[1])

            # Prunning
            if bestmove[1] > beta: return bestmove
            else: alpha = max(alpha,bestmove[1])

        return bestmove

    def minval(self, gamestate, agentIndex, depth, alpha, beta):
        bestmove = ("min",float("inf"))
        for action in gamestate.getLegalActions(agentIndex):
            succAction = (action,self.alphabeta(gamestate.generateSuccessor(agentIndex,action),
                                      (depth + 1)%gamestate.getNumAgents(),depth+1, alpha, beta))
            bestmove = min(bestmove,succAction,key=lambda x:x[1])

            # Prunning
            if bestmove[1] < alpha: return bestmove
            else: beta = min(beta, bestmove[1])

        return bestmove

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gamestate):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction
          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
          The expectimax function returns a tuple of (actions,
        """
        "*** YOUR CODE HERE ***"
        # calling expectimax with the depth we are going to investigate
        maxDepth = self.depth * gamestate.getNumAgents()
        return self.expectimax(gamestate, "expect", maxDepth, 0)[0]

    def expectimax(self, gamestate, action, depth, agentIndex):

        if depth == 0 or gamestate.isLose() or gamestate.isWin():
            return (action, self.evaluationFunction(gamestate))

        # if pacman (max agent) - return max successor value
        if agentIndex == 0:
            return self.maxvalue(gamestate,action,depth,agentIndex)
        # if ghost (EXP agent) - return probability value
        else:
            return self.expvalue(gamestate,action,depth,agentIndex)

    def maxvalue(self,gamestate,action,depth,agentIndex):
        bestmove = ("max", -(float('inf')))
        for legalAction in gamestate.getLegalActions(agentIndex):
            nextAgent = (agentIndex + 1) % gamestate.getNumAgents()
            succAction = None
            if depth != self.depth * gamestate.getNumAgents():
                succAction = action
            else:
                succAction = legalAction
            succValue = self.expectimax(gamestate.generateSuccessor(agentIndex, legalAction),
                                        succAction,depth - 1,nextAgent)
            bestmove = max(bestmove,succValue,key = lambda x:x[1])
        return bestmove

    def expvalue(self,gamestate,action,depth,agentIndex):
        legalActions = gamestate.getLegalActions(agentIndex)
        averageScore = 0
        propability = 1.0/len(legalActions)
        for legalAction in legalActions:
            nextAgent = (agentIndex + 1) % gamestate.getNumAgents()
            bestmove = self.expectimax(gamestate.generateSuccessor(agentIndex, legalAction),
                                         action, depth - 1, nextAgent)
            averageScore += bestmove[1] * propability
        return (action, averageScore)

import math

def betterEvaluationFunction(currentgamestate):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).
      DESCRIPTION: <write something here so we know what you did>
      Evaluate state by  :
            * closest food
            * food left
            * capsules left
            * distance to ghost
    """
    "*** YOUR CODE HERE ***"

    # Useful information you can extract from a gamestate (pacman.py)
    newPos = currentgamestate.getPacmanPosition()
    newFood = currentgamestate.getFood().asList()

    minFoodist = float('inf')
    for food in newFood:
        minFoodist = min(minFoodist, manhattanDistance(newPos, food))

    ghostDist = 0
    for ghost in currentgamestate.getGhostPositions():
        ghostDist = manhattanDistance(newPos, ghost)
        if (ghostDist < 2):
            return -float('inf')

    foodLeft = currentgamestate.getNumFood()
    capsLeft = len(currentgamestate.getCapsules())

    foodLeftMultiplier = 950050
    capsLeftMultiplier = 10000
    foodDistMultiplier = 950

    additionalFactors = 0
    if currentgamestate.isLose():
        additionalFactors -= 50000
    elif currentgamestate.isWin():
        additionalFactors += 50000

    return 1.0/(foodLeft + 1) * foodLeftMultiplier + ghostDist + \
           1.0/(minFoodist + 1) * foodDistMultiplier + \
           1.0/(capsLeft + 1) * capsLeftMultiplier + additionalFactors

# Abbreviation
better = betterEvaluationFunction