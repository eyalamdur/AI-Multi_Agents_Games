from Agent import Agent, AgentGreedy
from WarehouseEnv import WarehouseEnv, manhattan_distance
from enum import Enum
import random
import time
from func_timeout import func_timeout, FunctionTimedOut

BATTERY_WEIGHT = 1000
CRITICAL_CHARGER_WEIGHT = 1000
CREDIT_WEIGHT = 1000
PACKAGE_WEIGHT = 50
TIME_LIMITATION = 0.8
EXPECTIMAX_TIME_LIMITATION = 0.75

def smart_heuristic(env: WarehouseEnv, robot_id: int):
    # Get robot by id and validate it exists
    robot = env.get_robot(robot_id)
    # enemy_robot = env.get_robot(1-robot_id)
    if not robot:
        return 0
    
    # Calculate the target point for the robot and the closest package
    package = closest_package(env, robot_id)
    charger = closest_charger(env, robot_id)
    target = package.destination if robot.package else package.position
        
    # distances calculation
    target_distance = manhattan_distance(target, robot.position)
    charger_distance = manhattan_distance(charger.position, robot.position)
    
    # Optional weights for the heuristic calculation And additional cost for the package
    credit_weight, additional_cost, package_weight = (CREDIT_WEIGHT, 0, 10) if robot.package else (CREDIT_WEIGHT/10, manhattan_distance(package.position, package.destination), 0)
    #credits_gap = robot.credit - enemy_robot.credit
    
    # If i dont have credit to charge with, go to package
    if robot.credit <= 0:
        return package_weight*PACKAGE_WEIGHT + robot.credit * CREDIT_WEIGHT * CREDIT_WEIGHT - target_distance
        
    # If I'm in alarming battery situation
    if charger_distance == robot.battery and robot.credit > 0:
        return robot.battery * BATTERY_WEIGHT + robot.credit * credit_weight - charger_distance * CRITICAL_CHARGER_WEIGHT
    
    # If I have a package and the battery is not enough to deliver it, return to the charger
    if target_distance + manhattan_distance(charger.position, target) >= robot.battery and robot.credit > 0:
        return robot.battery * BATTERY_WEIGHT/10 + robot.credit * credit_weight - charger_distance
    
    # Otherwise, return the heuristic value
    return package_reward(package) * package_weight - target_distance - additional_cost + (robot.credit * credit_weight) + (robot.battery * BATTERY_WEIGHT)
    

# def smart_heuristic(env: WarehouseEnv, robot_id: int):
#     CREDIT_DIFFERENCE_WEIGHT = 100
#     BATTERY_WEIGHT = 100
#     DISTANCE_WEIGHT = 50
#     NUM_STEPS_WEIGHT = 100
#     PACKAGE_WEIGHT = 90

#     robot = env.get_robot(robot_id)
#     other_robot = env.get_robot(1-robot_id)

#     result = 0
#     credit_difference = robot.credit - other_robot.credit
#     result += CREDIT_DIFFERENCE_WEIGHT * credit_difference

#     if env.done():
#         return result

#     # if robot.battery == 0:
#     #     result -= NUM_STEPS_WEIGHT/env.num_steps
#     # else:
#     #     result += NUM_STEPS_WEIGHT/env.num_steps

#     closest_charge_station_distance = manhattan_distance(robot.position, get_closest_charge_station(env, robot_id).position)
#     if robot.battery == closest_charge_station_distance + 1 and robot.credit > 0:
#         result -= DISTANCE_WEIGHT * closest_charge_station_distance
#         result += BATTERY_WEIGHT * robot.battery

#     if robot.package is not None:
#         result += PACKAGE_WEIGHT + CREDIT_DIFFERENCE_WEIGHT * credit_difference
#         distance_from_destination = manhattan_distance(robot.package.destination, robot.position)
#         battery_cost = distance_from_destination + manhattan_distance(robot.package.destination, get_closest_charge_station(env, robot_id).position)
#         if battery_cost <= robot.battery:
#             # Reach the destination of the package.
#             result -= DISTANCE_WEIGHT * battery_cost
#         else:
#             # Go charge
#             result -= DISTANCE_WEIGHT * closest_charge_station_distance
#             result += BATTERY_WEIGHT * robot.battery
#     # No package
#     else:
#         existing_packages = [package for package in env.packages if package.on_board]
#         reachable_packages = [package for package in existing_packages if manhattan_distance(package.position, package.destination) + manhattan_distance(package.position, robot.position) <= robot.battery]
#         if not reachable_packages and robot.credit > 0:
#             result -= DISTANCE_WEIGHT * closest_charge_station_distance
#             result += BATTERY_WEIGHT * robot.battery
#         elif reachable_packages:
#             # No package, but can reach one.
#             most_valuable_package = max(reachable_packages, key=lambda package: manhattan_distance(package.position, package.destination))
#             result -= DISTANCE_WEIGHT * manhattan_distance(most_valuable_package.position, robot.position)
#             result += PACKAGE_WEIGHT * manhattan_distance(most_valuable_package.position, most_valuable_package.destination)
#     return result

# -------------------------------------- Agents ------------------------------------ #
    
class AgentGreedyImproved(AgentGreedy):
    def heuristic(self, env: WarehouseEnv, robot_id: int):
        return smart_heuristic(env, robot_id)


class AgentMinimax(Agent):
    def init(self):
        self.best_op = None
    
    def minimax(self, env: WarehouseEnv, robot_id, depth, my_turn: bool, finish_time):
        # Case time finish or final state or depth limit
        if time.time() >= finish_time or depth == 0:
            return smart_heuristic(env, robot_id), env.get_legal_operators(robot_id)[0]
        
        curr_robot = robot_id if my_turn else 1 - robot_id
        ops, children = self.successors(env, curr_robot)
        chosen_value = float("-inf") if my_turn else float("inf")
        chosen_op = ops[0]

        # Choosing the most fitted value, according to minimax starategy
        for op, child in zip(ops, children):
            value, _ = func_timeout(timeout=finish_time-time.time(), func=self.minimax, args=(child, robot_id, depth-1, not my_turn, finish_time),)

            if my_turn and value > chosen_value:
                chosen_value = value
                chosen_op = op
            elif not my_turn and value < chosen_value:
                chosen_value = value
                chosen_op = op

        return chosen_value, chosen_op

    def run_step(self, env: WarehouseEnv, agent_id, time_limit):
        # Call minimax with the specified time limit for each invocation
        finish_time = time.time() + TIME_LIMITATION*time_limit
        try:
            func_timeout(timeout=finish_time-time.time(), func=self.run_minimax, args=(env, agent_id, finish_time),)
        except FunctionTimedOut:
            # in case of timeout and the best_op is None, choose the first legal operator
            if self.best_op is None:
                self.best_op = env.get_legal_operators(agent_id)[0]
                
        return self.best_op

    def run_minimax(self, env: WarehouseEnv, agent_id, finish_time):
        depth = 1
        max_value = float("-inf")
        while True:
            value, op = func_timeout(timeout=finish_time-time.time(), func=self.minimax, args=(env, agent_id, depth, True, finish_time),)
            if max_value < value and op in env.get_legal_operators(agent_id):
                max_value = value
                self.best_op = op
            depth += 1


class AgentAlphaBeta(Agent):
    def init(self):
        self.best_op = None
    
    def ABminimax(self, env: WarehouseEnv, robot_id, depth, my_turn: bool, alpha: float, beta: float):
        # Case time finish or final state or depth limit
        if depth == 0:
            return smart_heuristic(env, robot_id), env.get_legal_operators(robot_id)[0]

        curr_robot = robot_id if my_turn else 1 - robot_id
        ops, children = self.successors(env, curr_robot)
        chosen_op = ops[0]
        if my_turn:
            chosen_value = float("-inf")
            for op, child in zip(ops, children):
                value, _ = self.ABminimax(child, robot_id, depth - 1, not my_turn, alpha, beta)
                if chosen_value < value:
                    chosen_value = value
                    chosen_op = op
                alpha = max(chosen_value, alpha)
                if chosen_value >= beta:
                    return float("inf"), op
            return chosen_value, chosen_op
        # not my turn
        else:
            chosen_value = float("inf")
            for op, child in zip(ops, children):
                value, _ = self.ABminimax(child, robot_id, depth - 1, not my_turn, alpha, beta)
                if chosen_value > value:
                    chosen_value = value
                    chosen_op = op
                beta = min(chosen_value, beta)
                if chosen_value <= alpha:
                    return float("-inf"), op
            return chosen_value, chosen_op

    def run_step(self, env: WarehouseEnv, agent_id, time_limit):
        # Call minimax with the specified time limit for each invocation
        time_limit_ensure = TIME_LIMITATION*time_limit
        try:
            func_timeout(time_limit_ensure, self.run_ABminimax, args=(env, agent_id))
        except FunctionTimedOut:
            # in case of timeout and the best_op is None, choose the first legal operator
            if self.best_op is None:
                self.best_op = env.get_legal_operators(agent_id)[0]
                
        return self.best_op
    
    def run_ABminimax(self, env: WarehouseEnv, agent_id):
        depth = 1
        max_value = float("-inf")
        while True:
            value, op = self.ABminimax(env, agent_id, depth, True, float("-inf"), float("inf"))
            if max_value < value and op in env.get_legal_operators(agent_id):
                max_value = value
                self.best_op = op
            depth += 1
        
class AgentExpectimax(Agent):
    def __init__(self):
        self.special_ops = ["move north", "pick_up"]

    def run_step(self, env, agent_index, time_limit):
        # Run the Expectimax algorithm to get the best move
        finish_time = time.time() + time_limit * EXPECTIMAX_TIME_LIMITATION
        depth = 1
        max_value, best_op = float("-inf"), env.get_legal_operators(agent_index)[0]
        
        while time.time() < finish_time:
            value, op = self.expectimax(env, agent_index, finish_time, depth, my_turn=True)
            if value > max_value:
                max_value, best_op = value, op
            depth += 1

        return best_op

    def expectimax(self, env, robot_id, time_finish, depth, my_turn):
        # Check if the search should be finished and return the heuristic value
        if self.finish_search(env, time_finish, depth):
            return smart_heuristic(env, robot_id), env.get_legal_operators(robot_id)[0]

        # Get the children of the current state and their operators
        current_robot = robot_id if my_turn else 1-robot_id
        ops, children = self.successors(env, current_robot)
        chosen_value, chosen_op = float("-inf") if my_turn else float("inf"), ops[0]

        # Choosing the most fitted value, according to expectimax Astrategy
        if my_turn:
            chosen_value, chosen_op = self.max_value(children, ops, robot_id, time_finish, depth, my_turn, chosen_value)
        else:
            chosen_value, chosen_op = self.expect_value(children, ops, robot_id, time_finish, depth, my_turn)

        return chosen_value, chosen_op

    # Check if the search should be finished time limit, depth limit or both robots are out of battery
    def finish_search(self, env, time_finish, depth):
        FIRST_ROBOT_ID, SECOND_ROBOT_ID = 0, 1
        if time.time() >= time_finish or depth == 0 or \
            (env.get_robot(FIRST_ROBOT_ID).battery == 0 and env.get_robot(SECOND_ROBOT_ID).battery == 0):
            return True
        return False
    
    # Calculate and returns expect of the value of the children
    def expect_value(self, children, ops, robot_id, time_finish, depth, my_turn):
        values_sum = 0
        num_of_ops = len(ops)
        for op, child in zip(ops, children):
            value, _ = self.expectimax(child, robot_id, time_finish, depth-1, not my_turn)
            values_sum += value
            # If the operator is special, give it double probability
            if op in self.special_ops:
                values_sum , num_of_ops = values_sum + value, num_of_ops + 1
        return values_sum / num_of_ops, ops[0]
        
    
    # Calculate and returns the max value of the children
    def max_value(self, children, ops, robot_id, time_finish, depth, my_turn, current_value):
        for op, child in zip(ops, children):
            value, _ = self.expectimax(child, robot_id, time_finish, depth-1, not my_turn)
            if value > current_value:
                current_value, chosen_op = value, op
        return current_value, chosen_op
    
    
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
