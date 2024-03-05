import numpy as np
import matplotlib.pyplot as plt

def plot_angle_distribution_polar(vectors, normal_vector):
    """
    Plot the distribution of angles between a series of 3D vectors and a given plane,
    represented by its normal vector, in polar coordinates.
    
    Parameters:
    - vectors: numpy.ndarray of shape (n, 3), where n is the number of vectors.
    - normal_vector: numpy.ndarray of shape (3,), representing the normal vector of the plane.
    """
    # Normalize the normal vector to ensure it's a unit vector
    normal_vector = normal_vector / np.linalg.norm(normal_vector)
    
    # Calculate the projection of the vectors onto the plane
    projection = vectors - np.dot(vectors, normal_vector)[:,None] * normal_vector
    projection /= np.linalg.norm(projection, axis=1)[:, np.newaxis]

    # Choose a random basis vector in the plane to calculate the angles
    basis_vector = np.random.randn(3)
    basis_vector -= np.dot(basis_vector, normal_vector) * normal_vector
    basis_vector /= np.linalg.norm(basis_vector)
    basis_vector2 = np.cross(normal_vector, basis_vector)

    # Calculate the angles between the projected vectors and the basis vector
    angles = np.arccos(np.dot(projection, basis_vector))

    # Convert angles to degrees for easier interpretation
    angles_deg = np.degrees(angles)
    
    # Add 180 degrees to negative angles to ensure the distribution is symmetric
    y = np.dot(projection, basis_vector2)
    angles_deg = np.where(y < 0, 360 - angles_deg, angles_deg)

    # Plot histogram of angles in polar coordinates
    plt.figure(figsize=(8, 8))
    ax = plt.subplot(111, polar=True)
    
    # Compute histogram
    hist, bins = np.histogram(angles_deg, bins=np.linspace(0, 360, 36))
    
    # Compute width of each bin
    width = np.diff(bins)
    
    # Plot
    bars = ax.bar(np.radians(bins[:-1]), hist, width=np.radians(width), bottom=0.0)
    
    ax.set_theta_zero_location("N")  # Set the direction of 0 degrees to the top
    ax.set_theta_direction(-1)  # Set the direction of degrees to be clockwise
    ax.set_title("Distribution of Angles with Respect to the Given Plane")
    
    plt.show()
    return basis_vector, basis_vector2, angles_deg

# Example usage:
# Generate random 3D unit vectors for demonstration
np.random.seed(42)
n_vectors = 10000
vectors = np.random.randn(n_vectors, 3)
vectors /= np.linalg.norm(vectors, axis=1)[:, np.newaxis]  # Normalize to unit length

# Normal vector to the xy-plane (z-axis)
normal_vector = np.array([0, 0, 1])

# Plot the angle distribution
plot_angle_distribution_polar(vectors, normal_vector)
