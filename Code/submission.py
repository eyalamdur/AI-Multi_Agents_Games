from Agent import Agent, AgentGreedy
from WarehouseEnv import WarehouseEnv, manhattan_distance
from enum import Enum
import random
import time

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
    
    def gap_credit_points(self, env: WarehouseEnv, robot_id):
        my_robot = env.get_robot(robot_id)
        foe_robot = env.get_robot(abs(robot_id-1))
        return my_robot.credit - foe_robot.credit
    
    def minimax(self, env: WarehouseEnv, robot_id, time_finish, depth, my_turn: bool):
        # Case time finish or final state or depth limit
        if time.time() >= time_finish or depth == 0 or\
                (env.get_robot(robot_id).battery == 0 and env.get_robot(abs(robot_id-1)).battery == 0):
            return smart_heuristic(env, robot_id), None

        ops, children = self.successors(env, robot_id)
        chosen_value = float("-inf") if my_turn else float("inf")
        chosen_op = None

        # Choosing the most fitted value, according to minimax astategy
        for op, child in zip(ops, children):
            value, _ = self.minimax(child, robot_id, time_finish, depth-1, not my_turn)
            if my_turn and value > chosen_value:
                chosen_value = value
                chosen_op = op
            elif not my_turn and value < chosen_value:
                chosen_value = value
                chosen_op = op
            if time.time() >= time_finish:
                break
        return chosen_value, chosen_op

    def run_step(self, env: WarehouseEnv, agent_id, time_limit):
        finish_time = time.time() + time_limit
        depth = 1
        while time.time() < finish_time:
            _, op = self.minimax(env, agent_id, finish_time, depth, True)
            depth += 1
        return op

class AgentAlphaBeta(Agent):
    # TODO: section c : 1
    def ABminimax(self, env: WarehouseEnv, robot_id, time_finish, depth, my_turn: bool, alpha: float, beta: float):
        # Case time finish or final state or depth limit
        if time.time() >= time_finish or depth == 0 or \
                (env.get_robot(robot_id).battery == 0 and env.get_robot(abs(robot_id - 1)).battery == 0):
            return smart_heuristic(env, robot_id), None

        ops, children = self.successors(env, robot_id)
        chosen_value = float("-inf") if my_turn else float("inf")
        chosen_op = None

        # Choosing the most fitted value, according to ABminimax astategy
        for op, child in zip(ops, children):
            value, _ = self.ABminimax(child, robot_id, time_finish, depth - 1, not my_turn, alpha, beta)
            if my_turn:
                chosen_value = value if value > chosen_value else chosen_value
                alpha = chosen_value if chosen_value > alpha else alpha
                chosen_op = op
                if chosen_value >= beta:
                    return float("inf"), op
            else:
                chosen_value = value if value < chosen_value else chosen_value
                beta = chosen_value if chosen_value < beta else beta
                chosen_op = op
                if chosen_value <= alpha:
                    return float("-inf"), op
            if time.time() >= time_finish:
                break
        return chosen_value, chosen_op

    def run_step(self, env: WarehouseEnv, agent_id, time_limit):
        finish_time = time.time() + time_limit
        depth = 1
        while time.time() < finish_time:
            _, op = self.ABminimax(env, agent_id, finish_time, depth, True, float("-inf"), float("inf"))
            depth += 1
        return op

class AgentExpectimax(Agent):
    def __init__(self):
        pass

    def run_step(self, env, agent_index, time_limit):
        # Run the Expectimax algorithm to get the best move
        finish_time = time.time() + time_limit * 0.8
        depth = 1
        while time.time() < finish_time:
            _, op = self.expectimax(env, agent_index, finish_time, depth, True)
            depth += 1
        return op

    def expectimax(self, env, robot_id, time_finish, depth, my_turn):
        # Check if the search should be finished and return the heuristic value
        if self.finish_search(env, time_finish, depth):
            return (smart_heuristic(env, robot_id) if my_turn else (-smart_heuristic(env, robot_id))), None

        # Get the children of the current state and their operators
        ops, children = self.successors(env, robot_id)
        chosen_value, chosen_op= float("-inf") if my_turn else float("inf"), None

        # Choosing the most fitted value, according to expectimax Astrategy
        if my_turn:
            chosen_value, chosen_op = self.max_value(children, ops, robot_id, time_finish, depth, my_turn, chosen_value)
        else:
            chosen_value = self.expect_value(children, robot_id, time_finish, depth, my_turn)

        return chosen_value, chosen_op

    # Check if the search should be finished time limit, depth limit or both robots are out of battery
    def finish_search(self, env, time_finish, depth):
        FIRST_ROBOT_ID, SECOND_ROBOT_ID = 0, 1
        if time.time() >= time_finish or depth == 0 or \
            (env.get_robot(FIRST_ROBOT_ID).battery == 0 and env.get_robot(abs(SECOND_ROBOT_ID)).battery == 0):
            return True
        return False
    
    # Calculate and returns expect of the value of the children
    def expect_value(self, children, robot_id, time_finish, depth, my_turn):
        values_sum = sum([self.expectimax(child, 1 - robot_id, time_finish, depth-1, not my_turn)[0] for child in children])
        return values_sum / len(children)
    
    # Calculate and returns the max value of the children
    def max_value(self, children, ops, robot_id, time_finish, depth, my_turn, current_value):
        for op, child in zip(ops, children):
            value, _ = self.expectimax(child, 1 - robot_id, time_finish, depth-1, not my_turn)
            if value > current_value:
                current_value, chosen_op = value, op
            if time.time() >= time_finish:
                break
        return current_value, chosen_op
    
# ------------------------ smart_heuristic Helper Functions ------------------------ #
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
