import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import LineString


def check_self_intersection(coordinates):
    coordinates = np.vstack(coordinates)
    line = LineString(coordinates)
    for i in range(len(line.coords) - 1):
        segment = line.coords[i:i+2]
        segment_line = LineString(segment)
        if line.intersects(segment_line):
            return True

    return False


class TrajectoryGenerator:
    def __init__(self, position_range=[0, 80], acceleration_range=[-4, 4], velocity_range=[-4, 4], noise_level=1,
                 dimension=3):
        '''
        position_range: [pos_min, pos_max]
        acceleration_range: [acc_min, acc_max]
        velocity_range: [vel_min, vel_max]
        noise_level:
        '''
        self.position_range = position_range
        self.acceleration_range = acceleration_range
        self.velocity_range = velocity_range
        self.noise_level = noise_level
        self.dimension = dimension


    def recalibration(self, position, velocity):
        acceleration = np.random.uniform(self.acceleration_range[0], self.acceleration_range[1], self.dimension)
        for d in range(self.dimension):
            if position[d] <= self.position_range[0] or position[d] > self.position_range[1]:
                velocity[d] = -velocity[d]
                acceleration[d] = -acceleration[d] if np.sign(acceleration[d] + velocity[d]) != np.sign(velocity[d]) else acceleration[d]
            velocity[d] = velocity[d] + acceleration[d]
            velocity[d] = np.clip(velocity[d], self.velocity_range[0], self.velocity_range[1])
            position[d] = np.clip(position[d], self.position_range[0], self.position_range[1])
        return position, velocity


    def generate_trajectory(self, num_steps):
        trajectory = []
        # 1) Init the position, velosity and acceleration
        current_position = np.zeros(self.dimension)
        current_velocity = np.random.uniform(0, self.velocity_range[1], self.dimension)
        # 2) Update velocity based on acceleration
        current_acceleration = np.random.uniform(self.acceleration_range[0], self.acceleration_range[1], self.dimension)
        current_velocity += current_acceleration
        current_acceleration = np.random.uniform(self.acceleration_range[0], self.acceleration_range[1], self.dimension)
        trajectory.append(current_position.copy())

        for _ in range(num_steps - 1):
            '''
            # Add random angle to the velocity direction
            if angle is None:
                angle = np.random.uniform(-np.pi / 4, np.pi / 4)  # Adjust the angle range as needed
            rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)],
                                        [np.sin(angle), np.cos(angle)]])
            current_velocity = np.dot(rotation_matrix, current_velocity)
            '''
            # 3) Update position based on velocity
            new_position = current_position + current_velocity
            #new_position, new_velocity = self.recalibration(new_position, current_velocity)
            # 4) Add noise to the position
            noise = np.random.normal(0, self.noise_level, self.dimension)  # Adjust the noise level
            new_position = new_position + noise
            new_position, new_velocity = self.recalibration(new_position, current_velocity)
            trajectory.append(new_position.copy())
            current_velocity = new_velocity
            current_position = new_position

        return np.array(trajectory)


    def generate_data(self, num_seq , num_steps, prefix=None, save=True):
        data = []
        for i in range(num_seq):
            seq = self.generate_trajectory(num_steps)
            data.append(seq)
        data = np.array(data, dtype=float)
        if prefix is not None:
            store_dir = f"{prefix}_p{self.position_range[1]}_v{self.velocity_range[1]}_a{self.acceleration_range[1]}_n{num_seq}_t{num_steps}.npy"
        else:
            store_dir = f"p{self.position_range[1]}_v{self.velocity_range[1]}_a{self.acceleration_range[1]}_n{num_seq}_t{num_steps}.npy"
        if save:
            np.save(store_dir, data)

    def plot_trajectory(self, trajectory):
        x = trajectory[:, 0]
        y = trajectory[:, 1]
        plt.plot(x, y, 'b-')
        plt.scatter(x, y, c='r', marker='o')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Trajectory')
        plt.grid(True)
        plt.show()


    def plot_trajectory_3d(self, trajectory):
        x = trajectory[:, 0]
        y = trajectory[:, 1]
        z = trajectory[:, 2]

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(x, y, z, 'b-')
        ax.scatter(x, y, z, c='r', marker='o')

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('3D Trajectory')
        plt.show()


# Example usage
generator = TrajectoryGenerator(position_range=[0, 70], acceleration_range=[-0.5, 0.5], velocity_range=[-5, 5], noise_level=0.8, dimension=3)
trajectory = generator.generate_trajectory(num_steps=50) # np: (num_steps, dimension)
generator.plot_trajectory_3d(trajectory)
generator.generate_data(num_seq=10, num_steps=50)