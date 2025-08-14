import numpy as np
from scipy.spatial.transform import Rotation
from scipy.optimize import minimize
import logging
import os
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO)

def hand_eye_calibration(T_WR: list[np.ndarray], T_CC: list[np.ndarray]) -> np.ndarray:
    """
    Perform hand-eye calibration to find the static transformation T_RC.

    Parameters:
        T_WR (list[np.ndarray]): List of robot poses in the world frame (4x4 matrices).
        T_CC (list[np.ndarray]): List of camera poses relative to its start frame (4x4 matrices).

    Returns:
        np.ndarray: Static transformation T_RC (4x4 matrix).
    """
    def objective(x: np.ndarray) -> float:
        # Convert x (flattened transformation) into a 4x4 matrix
        R = Rotation.from_rotvec(x[:3]).as_matrix()  # First 3 elements are rotation (rodrigues vector)
        t = x[3:]  # Last 3 elements are translation
        T_RC = np.eye(4)
        T_RC[:3, :3] = R
        T_RC[:3, 3] = t

        # Compute the error for all pose pairs
        error = 0.0
        for T_WR_i, T_CC_i in zip(T_WR, T_CC):
            # Compute AX - XB
            error_matrix = np.linalg.inv(T_RC) @ T_WR_i @ T_RC - T_CC_i
            rotation_error = np.linalg.norm(error_matrix[:3, :3], 'fro') ** 2  # Frobenius norm for rotation
            translation_error = np.linalg.norm(error_matrix[:3, 3]) ** 2       # Euclidean norm for translation
            error += rotation_error + translation_error

        return error

    # Initial guess for T_RC: identity matrix
    initial_guess = np.zeros(6)  # 3 for rotation (rodrigues), 3 for translation

    # Minimize the objective function
    result = minimize(objective, initial_guess, method='BFGS')

    if not result.success:
        logging.warning("Optimization did not converge: %s", result.message)
    else:
        logging.info("Optimization successful: %s", result.message)
    logging.info("Optimization result: %s", result)

    # Construct the final T_RC matrix
    R_final = Rotation.from_rotvec(result.x[:3]).as_matrix()
    t_final = result.x[3:]
    T_RC_final = np.eye(4)
    T_RC_final[:3, :3] = R_final
    T_RC_final[:3, 3] = t_final

    return T_RC_final

def load_transformations_from_directory(directory_path: str) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """
    Load transformation matrices from .npy files in the specified directory.

    Parameters:
        directory_path (str): Path to the base directory containing 'rob_poses' and 'zed_poses' subdirectories.

    Returns:
        tuple: Two lists of np.ndarray representing T_WR and T_CC.
    """
    T_WR = []
    T_CC = []

    # Load robot poses from 'rob_poses' subdirectory
    rob_poses_path = os.path.join(directory_path, 'rob_poses')
    rob_files = sorted([f for f in os.listdir(rob_poses_path) if f.endswith('.npy')])
    for file in rob_files:
        file_path = os.path.join(rob_poses_path, file)
        transformation = np.load(file_path)
        T_WR.append(transformation)

    # Load camera poses from 'zed_poses' subdirectory
    zed_poses_path = os.path.join(directory_path, 'zed_poses')
    zed_files = sorted([f for f in os.listdir(zed_poses_path) if f.endswith('.npy')])
    for file in zed_files:
        file_path = os.path.join(zed_poses_path, file)
        transformation = np.load(file_path)
        T_CC.append(transformation)

    return T_WR, T_CC

def plot_transformations(T_WR: list[np.ndarray], T_CC: list[np.ndarray], T_RC: np.ndarray):
    """
    Plot the transformations for visualization.

    Parameters:
        T_WR (list[np.ndarray]): List of robot poses in the world frame (4x4 matrices).
        T_CC (list[np.ndarray]): List of camera poses relative to its start frame (4x4 matrices).
        T_RC (np.ndarray): Static transformation T_RC (4x4 matrix).
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot every 10th robot pose
    for i, T in enumerate(T_WR):
        if i % 10 == 0:
            ax.scatter(T[0, 3], T[1, 3], T[2, 3], c='r', marker='o')
            ax.quiver(T[0, 3], T[1, 3], T[2, 3], T[0, 2], T[1, 2], T[2, 2], color='r', length=0.1)

    # Plot every 10th camera pose
    for i, T in enumerate(T_CC):
        if i % 10 == 0:
            ax.scatter(T[0, 3], T[1, 3], T[2, 3], c='b', marker='^')
            ax.quiver(T[0, 3], T[1, 3], T[2, 3], T[0, 2], T[1, 2], T[2, 2], color='b', length=0.1)

    # Plot every 10th transformed camera pose
    for i, T in enumerate(T_WR):
        if i % 10 == 0:
            T_transformed = T @ T_RC
            ax.scatter(T_transformed[0, 3], T_transformed[1, 3], T_transformed[2, 3], c='g', marker='x')
            ax.quiver(T_transformed[0, 3], T_transformed[1, 3], T_transformed[2, 3], T_transformed[0, 0], T_transformed[1, 0], T_transformed[2, 0], color='g', length=0.1)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()

# Example Usage:
directory_path = "/home/pbrick/zed2i/250107_Test_Onboard_500_3_0.05_20_1"
T_WR, T_CC = load_transformations_from_directory(directory_path)
P = np.array([[0, 0, 1, 0],
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 0, 1]
            ])
T_WR = [a @ P for a in T_WR]
T_RC = hand_eye_calibration(T_WR, T_CC)
print("Estimated T_RC:\n", T_RC)

# Plot the transformations
plot_transformations(T_WR, T_CC, T_RC)