import time
from Agent import Agent, AgentGreedy
from WarehouseEnv import WarehouseEnv, manhattan_distance
import random

BATTERY_WEIGHT = 1000
CREDIT_WEIGHT = 1000
TIME_LIMITATION = 0.2

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
        finish_time = time.time() + TIME_LIMITATION
        depth = 1
        while time.time() < finish_time:
            _, op = self.ABminimax(env, agent_id, finish_time, depth, True, float("-inf"), float("inf"))
            depth += 1
        return op

class AgentExpectimax(Agent):
    # TODO: section d : 1
    def run_step(self, env: WarehouseEnv, agent_id, time_limit):
        raise NotImplementedError()


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
    
    
    
    # ----------------- Helper Functions ----------------- #
def closest_package(env: WarehouseEnv, robot_id: int):
    robot = env.get_robot(robot_id)
    
    # If the robot is already carrying a package, return it
    if robot.package:
        return robot.package
    
    # Calculate the distance to each package and return the closest one
    package0_dist = manhattan_distance(env.packages[0].position, robot.position)
    package1_dist = manhattan_distance(env.packages[1].position, robot.position)
    return env.packages[1] if package0_dist > package1_dist else env.packages[0]

def closest_charger(env: WarehouseEnv, robot_id: int):
    robot = env.get_robot(robot_id)
    
    # Calculate the distance to each charger and return the closest one
    charger0_dist = manhattan_distance(env.charge_stations[0].position, robot.position)
    charger1_dist = manhattan_distance(env.charge_stations[1].position, robot.position)
    return env.charge_stations[1] if charger0_dist > charger1_dist else env.charge_stations[0]

def package_reward(package):
    # Calculate the reward for delivering the package
    return 2 * manhattan_distance(package.position, package.destination)
    
