from Agent import Agent, AgentGreedy
from WarehouseEnv import WarehouseEnv, manhattan_distance
import random
import time

TIME_LIMITATION = 0.8

def smart_heuristic(env: WarehouseEnv, robot_id: int):  
    BATTERY_WEIGHT = 110
    CREDIT_WEIGHT = 100
    PACKAGE_WEIGHT = 50
    DISTANCE_WEIGHT = 30
    # Get robots 
    robot = env.get_robot(robot_id)
    enemy_robot = env.get_robot(1-robot_id)
    
    # Define current state variables
    package = closest_package(env, robot_id)
    charger = closest_charger(env, robot_id)
    target = package.destination if robot.package else package.position
    
    # distances calculation
    target_distance = manhattan_distance(target, robot.position)
    charger_distance = manhattan_distance(charger.position, robot.position)
    
    # Pre-calculation for the heuristic
    credit_gap = robot.credit - enemy_robot.credit
    battery_cost = target_distance + manhattan_distance(charger.position, target) if robot.package else target_distance + manhattan_distance(package.position, package.destination) + manhattan_distance(package.destination, charger.position)
    package_weight = PACKAGE_WEIGHT if robot.package else PACKAGE_WEIGHT/10
    
    # Initial heuristic value (Add the credit difference and battery state to the heuristic)
    h_value = credit_gap * CREDIT_WEIGHT**2 + robot.battery * BATTERY_WEIGHT*2

    # If i dont have credit to charge with, go to package
    if robot.credit <= 0:
        h_value += package_reward(package) * package_weight + robot.credit * CREDIT_WEIGHT - target_distance * DISTANCE_WEIGHT
    
    # If I'm in critical battery situation
    if robot.battery == charger_distance and robot.credit > 0:
        h_value += BATTERY_WEIGHT * robot.battery - charger_distance * DISTANCE_WEIGHT

    # If I have enough battery to deliver the package and go back to the charger, do it
    if battery_cost <= robot.battery:
        h_value += package_reward(package) * package_weight + robot.credit * CREDIT_WEIGHT - battery_cost * DISTANCE_WEIGHT
    elif robot.credit > 0:   # Go charge (Only if I have credit)
        h_value += robot.battery * BATTERY_WEIGHT/3 - charger_distance * DISTANCE_WEIGHT

    return h_value
# -------------------------------------- Agents ------------------------------------ #
    
class AgentGreedyImproved(AgentGreedy):
    def heuristic(self, env: WarehouseEnv, robot_id: int):
        return smart_heuristic(env, robot_id)


class AgentMinimax(Agent):
    
    def minimax(self, env: WarehouseEnv, robot_id, time_finish, depth, my_turn: bool):
        # Case time finish or final state or depth limit
        if time.time() >= time_finish:
            raise TimeoutError
        if depth == 0:
            return smart_heuristic(env, robot_id), env.get_legal_operators(robot_id)[0]
        
        curr_robot = robot_id if my_turn else 1 - robot_id
        ops, children = self.successors(env, curr_robot)
        chosen_value = float("-inf") if my_turn else float("inf")
        chosen_op = ops[0]

        # Choosing the most fitted value, according to minimax starategy
        for op, child in zip(ops, children):
            value, _ = self.minimax(child, robot_id, time_finish, depth-1, not my_turn)

            if my_turn and value > chosen_value:
                chosen_value = value
                chosen_op = op
            elif not my_turn and value < chosen_value:
                chosen_value = value
                chosen_op = op
            if time.time() >= time_finish:
                raise TimeoutError
        return chosen_value, chosen_op

    def run_step(self, env: WarehouseEnv, agent_id, time_limit):
        finish_time = time.time() + TIME_LIMITATION*time_limit
        depth = 1
        best_op = env.get_legal_operators(agent_id)[0]
        try:
            while time.time() < finish_time and depth <= env.num_steps:
                _, best_op = self.minimax(env, agent_id, finish_time, depth, True)
                depth += 1
        except TimeoutError:
            pass
        return best_op


class AgentAlphaBeta(Agent):
    
    def ABminimax(self, env: WarehouseEnv, robot_id, time_finish, depth, my_turn: bool, alpha: float, beta: float):
        # Case time finish or final state or depth limit
        if time.time() >= time_finish:
            raise TimeoutError
        
        if depth == 0:
            return smart_heuristic(env, robot_id), env.get_legal_operators(robot_id)[0]

        curr_robot = robot_id if my_turn else 1 - robot_id
        ops, children = self.successors(env, curr_robot)
        chosen_op = ops[0]
        if my_turn:
            chosen_value = float("-inf")
            for op, child in zip(ops, children):
                value, _ = self.ABminimax(child, robot_id, time_finish, depth - 1, not my_turn, alpha, beta)
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
                value, _ = self.ABminimax(child, robot_id, time_finish, depth - 1, not my_turn, alpha, beta)
                if chosen_value > value:
                    chosen_value = value
                    chosen_op = op
                beta = min(chosen_value, beta)
                if chosen_value <= alpha:
                    return float("-inf"), op
            return chosen_value, chosen_op


    def run_step(self, env: WarehouseEnv, agent_id, time_limit):
        finish_time = time.time() + TIME_LIMITATION * time_limit
        depth = 1
        best_op = env.get_legal_operators(agent_id)[0]
        try:
            while time.time() < finish_time and depth <= env.num_steps:
                _, op = self.ABminimax(env, agent_id, finish_time, depth, True, float("-inf"), float("inf"))
                best_op = op if op in env.get_legal_operators(agent_id) else best_op
                depth += 1
        except TimeoutError:
            pass
        return best_op
        
class AgentExpectimax(Agent):
    def __init__(self):
        self.special_ops = ["move east", "pick_up"]

    def run_step(self, env, agent_index, time_limit):
        # Run the Expectimax algorithm to get the best move
        finish_time = time.time() + time_limit * TIME_LIMITATION
        depth = 1
        best_op = env.get_legal_operators(agent_index)[0]
        try:
            while time.time() < finish_time and depth <= env.num_steps:
                _, op = self.expectimax(env, agent_index, finish_time, depth, my_turn=True)
                best_op = op if op in env.get_legal_operators(agent_index) else best_op
                depth += 1
        except TimeoutError:
            pass
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
            chosen_value, _ = self.expect_value(children, ops, robot_id, time_finish, depth, my_turn)

        return chosen_value, chosen_op

    # Check if the search should be finished time limit, depth limit or both robots are out of battery
    def finish_search(self, env, time_finish, depth):
        FIRST_ROBOT_ID, SECOND_ROBOT_ID = 0, 1
        if depth == 0 or (env.get_robot(FIRST_ROBOT_ID).battery == 0 and env.get_robot(SECOND_ROBOT_ID).battery == 0):
            return True
        if time.time() >= time_finish:
            raise TimeoutError
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

    index = 1 if package0_dist > package1_dist else 0

    # foe is not blocking:
    foe = env.get_robot(1-robot_id)
    if foe.battery == 0 and (manhattan_distance(foe.position, env.packages[index].position) == 0
                             or manhattan_distance(foe.position, env.packages[index].destination) == 0):
        index = 1-index

    return env.packages[index]

# Helper function to get the closest charger to the robot
def closest_charger(env: WarehouseEnv, robot_id: int):
    robot = env.get_robot(robot_id)
    
    # Calculate the distance to each charger and return the closest one
    charger0_dist = manhattan_distance(env.charge_stations[0].position, robot.position)
    charger1_dist = manhattan_distance(env.charge_stations[1].position, robot.position)

    index = 1 if charger0_dist > charger1_dist else 0
    # foe is not blocking:
    foe = env.get_robot(1 - robot_id)
    if foe.battery == 0 and manhattan_distance(foe.position, env.charge_stations[index].position) == 0:
        index = 1 - index

    return env.charge_stations[index]

# Helper function to calculate the reward for delivering a package
def package_reward(package):
    return 2 * manhattan_distance(package.position, package.destination)
