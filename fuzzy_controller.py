import random
import numpy as np
import skfuzzy as fuzz

window_x = 720
window_y = 480
pixel_size = 10


def rotate_coordinates(coordinates, direction):
    window_size = np.array([window_x, window_y]) // pixel_size
    x, y = coordinates[0], coordinates[1]
    if direction == "UP":
        return coordinates
    elif direction == "DOWN":
        return window_size - coordinates - 1
    elif direction == "LEFT":
        coordinates[0] = window_size[1] - y - 1
        coordinates[1] = x
        return coordinates
    elif direction == "RIGHT":
        coordinates[0] = y
        coordinates[1] = window_size[0] - x - 1
        return coordinates


def rotate_board(game_state):
    direction = game_state["snake_direction"]
    for key, state in game_state.items():
        if key == "snake_direction":
            continue
        shape_len = len(state.shape)
        if shape_len == 1:
            game_state[key] = rotate_coordinates(state, direction)
        else:
            game_state[key] = np.apply_along_axis(func1d=rotate_coordinates, axis=1, arr=state, direction=direction)
    return game_state


def convert_pixels_to_grid_cords(coordinates):
    return coordinates // pixel_size


def fruit_rule(game_state):
    snake_position = game_state["snake_position"]
    fruit_position = game_state["fruit_position"]
    distance = fruit_position - snake_position

    dist_max = (max(window_x, window_y) // pixel_size)
    dy_range = np.arange(-dist_max, dist_max)
    dy_high = fuzz.smf(dy_range, -10, 0)
    dy_low = 1 - dy_high

    # dy_high -> trzeba zawrocic/skrecic
    # dy_low -> idziemy w strone owoca
    dy_level_low = fuzz.interp_membership(dy_range, dy_low, distance[1])
    dy_level_high = fuzz.interp_membership(dy_range, dy_high, distance[1])

    # dx_minus -> lewo
    # dx_plus -> prawo
    # dx_center -> do przodu
    zero_index = np.where(dy_range == 0)[0][0]
    dx_center = fuzz.gaussmf(dy_range, 0, 3)
    dx_plus = 1 - dx_center
    dx_plus[0:zero_index] = 0
    dx_minus = 1 - dx_center
    dx_minus[zero_index:] = 0

    dx_level_low = fuzz.interp_membership(dy_range, dx_minus, distance[0])
    dx_level_center = fuzz.interp_membership(dy_range, dx_center, distance[0])
    dx_level_high = fuzz.interp_membership(dy_range, dx_plus, distance[0])

    return {
        "dy_low": dy_level_low,
        "dy_high": dy_level_high,
        "dx_low": dx_level_low,
        "dx_center": dx_level_center,
        "dx_high": dx_level_high
    }


def walls_rule(game_state):
    snake_position = game_state["snake_position"]
    snake_direction = game_state["snake_direction"]

    if snake_direction == "LEFT" or snake_direction == "RIGHT":
        maximum = (window_x // pixel_size - 1)
    else:
        maximum = (window_y // pixel_size - 1)
    # y spore -> na wprost
    # y male -> skret
    # x male -> w prawo
    # x duze -> w lewo
    head_x_range = np.arange(0, maximum)
    wall_x_high = fuzz.smf(head_x_range, maximum - maximum * 0.2, maximum)
    wall_x_low = 1 - fuzz.smf(head_x_range, 0, maximum * 0.2)

    head_y_range = np.arange(0, 72)

    wall_y_low = 1 - fuzz.smf(head_y_range, 0, 10)
    wall_y_high = 1 - wall_y_low

    x_level_high = fuzz.interp_membership(head_x_range, wall_x_high, snake_position[0])
    x_level_low = fuzz.interp_membership(head_x_range, wall_x_low, snake_position[0])
    y_level_high = fuzz.interp_membership(head_y_range, wall_y_high, snake_position[1])
    y_level_low = fuzz.interp_membership(head_y_range, wall_y_low, snake_position[1])
    return {
        "wall_x_high": x_level_high,
        "wall_x_low": x_level_low,
        "wall_y_high": y_level_high,
        "wall_y_low": y_level_low
    }


def evaluate_rules(rules):
    dir_range = np.arange(0, 1, 0.01)
    dir_forward = fuzz.trimf(dir_range, [0.3, 0.5, 0.7])
    dir_left = fuzz.trimf(dir_range, [0, 0.1, 0.3])
    dir_right = fuzz.trimf(dir_range, [0.7, 0.9, 1.])

    # 1 rule: if dy_low and dx_center => dir_forward
    # 2 rule: if dy_high and dx_low => dir_left
    # 3 rule: if dy_high and dx_high => dir_right
    # 4 rule: if wall_y_low and wall_x_low => dir_right
    # 5 rule: if wall_y_low and wall_x_high => dir_left

    out_activations = []
    rule1 = np.fmin(rules["dy_low"], rules["dx_center"])
    out_activations.append(np.fmin(rule1, dir_forward))

    rule2 = np.fmin(rules["dy_high"], rules["dx_low"])
    out_activations.append(np.fmin(rule2, dir_left))

    rule3 = np.fmin(rules["dy_high"], rules["dx_high"])
    out_activations.append(np.fmin(rule3, dir_right))

    rule4 = np.fmin(rules["wall_y_low"], rules["wall_x_low"])
    out_activations.append(np.fmin(rule4, dir_right))

    rule5 = np.fmin(rules["wall_y_low"], rules["wall_x_high"])
    out_activations.append(np.fmin(rule5, dir_left))

    aggregated = np.fmax(out_activations[0], out_activations[1])
    for activation in out_activations[2:]:
        aggregated = np.fmax(aggregated, activation)
    out = fuzz.defuzz(dir_range, aggregated, 'centroid')
    return out


def avoid_wall(chosen_direction, snake_position):
    if chosen_direction == "UP" and snake_position[1] < 20:
        return random.choice(["LEFT", "RIGHT"])
    elif chosen_direction == "LEFT" and snake_position[0] < 20:
        return random.choice(["UP", "DOWN"])
    elif chosen_direction == "RIGHT" and snake_position[0] > window_x - 20:
        return random.choice(["UP", "DOWN"])
    elif chosen_direction == "DOWN" and snake_position[1] > window_y - 20:
        return random.choice(["LEFT", "RIGHT"])
    else:
        return chosen_direction


def will_collide_with_itself(snake_position, snake_body, direction):
    for body in snake_body[1:]:
        if body[1] == snake_position[1]:
            if direction == "RIGHT":
                if abs(snake_position[0] + 10 - body[0]) <= 15:
                    print(f'KOLIZJA {direction} - {snake_position} with {body}')
                    return True
            if direction == "LEFT":
                if abs(snake_position[0] - 10 - body[0]) <= 15:
                    print(f'KOLIZJA {direction} - {snake_position} with {body}')
                    return True
        elif body[0] == snake_position[0]:
            if direction == "UP":
                if abs(snake_position[1] - 10 - body[1]) <= 15:
                    print(f'KOLIZJA {direction} - {snake_position} with {body}')
                    return True
            if direction == "DOWN":
                if abs(snake_position[1] + 10 - body[1]) <= 15:
                    print(f'KOLIZJA {direction} - {snake_position} with {body}')
                    return True
    return False


def calculate_direction(snake_position, snake_direction, snake_body, fruit_position):
    # TODO
    game_state = {
        "snake_position": convert_pixels_to_grid_cords(np.array(snake_position)),
        "snake_direction": snake_direction,
        "snake_body": convert_pixels_to_grid_cords(np.array(snake_body)),
        "fruit_position": convert_pixels_to_grid_cords(np.array(fruit_position))
    }

    # Delta x i delta y - fruit
    game_state = rotate_board(game_state)

    rules = {}
    rules.update(fruit_rule(game_state))
    rules.update(walls_rule(game_state))
    try:
        out = evaluate_rules(rules)
    except AssertionError as e:
        print(e)
        out = 0.5

    directions = ['UP', 'RIGHT', 'DOWN', 'LEFT']
    direction_index = 0
    if out < 0.2:
        direction_index = 3
    elif out > 0.8:
        direction_index = 1

    chosen_direction = directions[(direction_index + directions.index(snake_direction)) % len(directions)]

    while will_collide_with_itself(snake_position, snake_body, chosen_direction):
        chosen_direction = random.choice([dir_candidate for dir_candidate in directions if dir_candidate != chosen_direction])
        print('Changed to: ',chosen_direction)
    chosen_direction = avoid_wall(chosen_direction, snake_position)
    return chosen_direction

# 11 5
