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
    distance = snake_position - fruit_position

    dist_max = (max(window_x, window_y) // pixel_size - 1)
    dy_range = np.arange(-dist_max, dist_max)
    dy_high = fuzz.smf(dy_range, 0, 10)
    dy_low = 1 - dy_high

    dy_level_low = fuzz.interp_membership(dy_range, dy_low, distance[1])
    dy_level_high = fuzz.interp_membership(dy_range, dy_high, distance[1])


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
    return random.choice(['UP', 'DOWN', 'LEFT', 'RIGHT'])

# 11 5
