import numpy as np

class ParticleFilter:
    def __init__(self, 
                 num_particles=1000, 
                 x_range=(0, 1000), 
                 y_range=(0, 1000),
                 motion_std=5.0,
                 measurement_std=3.0):

        self.num_particles = num_particles
        self.motion_std = motion_std
        self.measurement_std = measurement_std
        

        self.particles = np.empty((num_particles, 2))
        self.particles[:, 0] = np.random.uniform(x_range[0], x_range[1], size=num_particles)
        self.particles[:, 1] = np.random.uniform(y_range[0], y_range[1], size=num_particles)
        

        self.weights = np.ones(num_particles) / num_particles

    def predict(self, velocity=(0,0)):

        self.particles[:, 0] += velocity[0] + np.random.normal(0, self.motion_std, size=self.num_particles)
        self.particles[:, 1] += velocity[1] + np.random.normal(0, self.motion_std, size=self.num_particles)

    def update(self, measured_x, measured_y):

        dist_sq = (self.particles[:, 0] - measured_x)**2 + (self.particles[:, 1] - measured_y)**2
        

        self.weights = np.exp(-dist_sq / (2 * self.measurement_std**2))
        

        self.weights += 1e-300  
        self.weights /= np.sum(self.weights)

    def resample(self):

        cumulative_sum = np.cumsum(self.weights)
        cumulative_sum[-1] = 1.0 
        

        step = 1.0 / self.num_particles
        r = np.random.rand() * step
        positions = (r + np.arange(self.num_particles) * step)
        

        indices = np.searchsorted(cumulative_sum, positions)


        self.particles[:] = self.particles[indices]
        self.weights[:] = self.weights[indices]
        self.weights /= np.sum(self.weights)

    def estimate(self):

        x_est = np.average(self.particles[:, 0], weights=self.weights)
        y_est = np.average(self.particles[:, 1], weights=self.weights)
        return x_est, y_est
