import numpy as np

class EKFTracker:
    def __init__(self, tracker_id, initial_state, initial_covariance, process_noise, measurement_noise, max_miss_count=5):

        self.tracker_id = tracker_id  
        self.state = initial_state  
        self.covariance = initial_covariance
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        self.track_history = []  
        self.miss_count = 0 
        self.max_miss_count = max_miss_count  
        
    def predict(self, dt):

        F = np.array([[1, 0, dt, 0],
                      [0, 1, 0, dt],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]])
        Q = self.process_noise
        self.state = F @ self.state
        self.covariance = F @ self.covariance @ F.T + Q

    def update(self, measurement=None):

        if measurement is not None:
            H = np.array([[1, 0, 0, 0],
                          [0, 1, 0, 0]])
            R = self.measurement_noise
            S = H @ self.covariance @ H.T + R
            K = self.covariance @ H.T @ np.linalg.inv(S)

            y = measurement - H @ self.state
            self.state += K @ y
            self.covariance = (np.eye(len(self.state)) - K @ H) @ self.covariance
            self.miss_count = 0  
        else:
            self.miss_count += 1


        self.track_history.append(self.state[:2].tolist())
        if len(self.track_history) > 7:
            self.track_history.pop(0)

    def should_remove(self):

        return self.miss_count > self.max_miss_count

    def calculate_association_score(self, detection):

        predicted_position = self.state[:2]
        detected_position = np.array(detection['center'])
        distance = np.linalg.norm(predicted_position - detected_position)
        return distance

    def get_velocity_and_direction(self):

        vx, vy = self.state[2], self.state[3]
        speed = np.sqrt(vx**2 + vy**2)
        direction = np.arctan2(vy, vx)
        return speed, direction

    def get_track_history(self):

        return self.track_history