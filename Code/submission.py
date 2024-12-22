from Agent import Agent, AgentGreedy
from WarehouseEnv import WarehouseEnv, manhattan_distance
import random

BATTERY_WEIGHT = 100
CREDIT_WEIGHT = 1000

def smart_heuristic(env: WarehouseEnv, robot_id: int):
    # Get robot by id and validate it exists
    robot = env.get_robot(robot_id)
    if not robot:
        return 0
    
    # Calculate the target point for the robot and the closest package
    package = closest_package(env, robot_id)
    target = package.destination if robot.package else package.position
    if robot.package:
        cr_weight = 1000
    else:
        cr_weight = 100
    
    # If i have a package and the battery is not enough to deliver it, return to the charger
    if manhattan_distance(target, robot.position) >= robot.battery:
        return robot.battery * BATTERY_WEIGHT + (robot.credit + 1) * cr_weight - manhattan_distance(closest_charger(env, robot_id).position , robot.position)
    # Otherwise, return the heuristic value
    else:
        return package_reward(package) - manhattan_distance(robot.position, target) + (robot.credit+1) * cr_weight
    
    
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
    
