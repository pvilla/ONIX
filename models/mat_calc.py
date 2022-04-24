"""
    Support functions for the projection matrix calculation
"""

import numpy as np


def rot_x(theta):
    theta = np.deg2rad(theta)
    rot_matrix = np.array(
        [
            [1, 0, 0],
            [0, np.cos(theta), -np.sin(theta)],
            [0, np.sin(theta), np.cos(theta)],
        ]
    )
    return rot_matrix


def rot_y(theta):
    theta = np.deg2rad(theta)
    rot_matrix = np.array(
        [
            [np.cos(theta), 0, np.sin(theta)],
            [0, 1, 0],
            [-np.sin(theta), 0, np.cos(theta)],
        ]
    )
    return rot_matrix


def rot_z(theta):
    theta = np.deg2rad(theta)
    rot_matrix = np.array(
        [
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1],
        ]
    )
    return rot_matrix


def translation_mat(rot_matrix, theta):
    theta = np.deg2rad(theta)
    x_vec = rot_matrix * np.sin(theta)
    y_vec = 0
    z_vec = rot_matrix * np.cos(theta)
    trans_mat = np.array([[x_vec], [y_vec], [z_vec]])
    return trans_mat


def get_world_mat(theta_x, theta_y, theta_z):
    rot_mat = rot_z(theta_z) @ (rot_y(theta_y) @ rot_x(theta_x))
    trans_mat = translation_mat(1, theta_y)
    pad_mat = np.array([0, 0, 0, 1]).reshape(1, 4)
    world_mat = np.concatenate((rot_mat, trans_mat), 1)
    world_mat = np.concatenate((world_mat, pad_mat), 0)
    return world_mat
