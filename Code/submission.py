from Agent import Agent, AgentGreedy
from WarehouseEnv import WarehouseEnv, manhattan_distance
from enum import Enum
import random

BATTERY_WEIGHT = 1000
CREDIT_WEIGHT = 1000

def smart_heuristic(env: WarehouseEnv, robot_id: int):
    # Get robot by id and validate it exists
    robot = env.get_robot(robot_id)
    if not robot:
        return 0
    
    # Calculate the target point for the robot and the closest package
    package = closest_package(env, robot_id)
    target = package.destination if robot.package else package.position
    credit_weight, additional_cost = (CREDIT_WEIGHT, 0) if robot.package else (CREDIT_WEIGHT/10, manhattan_distance(package.position, package.destination))
    
    # If i have a package and the battery is not enough to deliver it, return to the charger
    if manhattan_distance(target, robot.position) + manhattan_distance(closest_charger(env, robot_id).position , target)  >= robot.battery and robot.credit != 0:
        return robot.battery * BATTERY_WEIGHT + robot.credit * credit_weight - manhattan_distance(closest_charger(env, robot_id).position , robot.position)
    # Otherwise, return the heuristic value
    else:
        return package_reward(package) - manhattan_distance(robot.position, target) - additional_cost + (robot.credit * credit_weight) + (robot.battery * BATTERY_WEIGHT)
    
    
class AgentGreedyImproved(AgentGreedy):
    def heuristic(self, env: WarehouseEnv, robot_id: int):
        return smart_heuristic(env, robot_id)


class AgentMinimax(Agent):
    # TODO: section b : 1
    def run_step(self, env: WarehouseEnv, agent_id, time_limit):
        raise NotImplementedError()


class AgentAlphaBeta(Agent):
    # TODO: section c : 1
    def run_step(self, env: WarehouseEnv, agent_id, time_limit):
        raise NotImplementedError()


class AgentExpectimax(Agent):
    def __init__(self, depth=5):
        self.depth = depth
        TURNS = {0: "My turn", 1: "Opponent's turn"}

    def run_step(self, env, agent_index, time_limit):
        # Run the Expectimax algorithm to get the best move
        operators = env.get_legal_operators(agent_index)
        children = self.successors(env, agent_index)
        
        for child in children:
            # calculate the value of each child and choose the one with the maximum value, return the operator            
        return (self.expectimax(env, agent_index, agent_index, self.depth, time_limit))

    def expectimax(self, env, agent_index, turn, depth, time_limit):
        # Stop conditions for the recursion
        if depth == 0 or env.done() or time_limit < 1:                          # HOW TO USE time_limit?
            return (self.evaluate(env, agent_index), None)
        
        children = self.successors(env, agent_index)
                
        # If its my turn calculate the maximum value of the children
        if turn == agent_index:
            current_max = float('-inf')
            for child in children:
                current_max = max(current_max, self.expectimax(child, agent_index, 1-turn, depth - 1, time_limit))
            
            return current_max
        
        # If its the opponent's turn calculate the average value of the children (Uniform distribution) 
        else:
            value = sum([self.expectimax(child, agent_index, 1-turn, depth - 1, time_limit) for child in children])
            return value / len(children)
    
    # Evaluation function on the current state
    def evaluate(self, env, agent_index):
        return env.get_balances()[agent_index] - env.get_balances()[1-agent_index]


# here you can check specific paths to get to know the environment
class AgentHardCoded(Agent):
    def __init__(self):
        self.step = 0
        # specifiy the path you want to check - if a move is illegal - the agent will choose a random move
        self.trajectory = ["move north", "move east", "move north", "move north", "pick_up", "move east", "move east",
                           "move south", "move south", "move south", "move south", "drop_off"]

    def run_step(self, env: WarehouseEnv, robot_id, time_limit):
        if self.step == len(self.trajectory):
            return self.run_random_step(env, robot_id, time_limit)
        else:
            op = self.trajectory[self.step]
            if op not in env.get_legal_operators(robot_id):
                op = self.run_random_step(env, robot_id, time_limit)
            self.step += 1
            return op

    def run_random_step(self, env: WarehouseEnv, robot_id, time_limit):
        operators, _ = self.successors(env, robot_id)

        return random.choice(operators)


# ------------------------ smart_heuristic Helper Functions ------------------------ #

#  Helper function to get the closest package to the robot
def closest_package(env: WarehouseEnv, robot_id: int):
    robot = env.get_robot(robot_id)
    
    # If the robot is already carrying a package, return it
    if robot.package:
        return robot.package
    
    # Calculate the distance to each package and return the closest one
    package0_dist = manhattan_distance(env.packages[0].position, robot.position)
    package1_dist = manhattan_distance(env.packages[1].position, robot.position)
    return env.packages[1] if package0_dist > package1_dist else env.packages[0]

# Helper function to get the closest charger to the robot
def closest_charger(env: WarehouseEnv, robot_id: int):
    robot = env.get_robot(robot_id)
    
    # Calculate the distance to each charger and return the closest one
    charger0_dist = manhattan_distance(env.charge_stations[0].position, robot.position)
    charger1_dist = manhattan_distance(env.charge_stations[1].position, robot.position)
    return env.charge_stations[1] if charger0_dist > charger1_dist else env.charge_stations[0]

# Helper function to calculate the reward for delivering a package
def package_reward(package):
    return 2 * manhattan_distance(package.position, package.destination)
