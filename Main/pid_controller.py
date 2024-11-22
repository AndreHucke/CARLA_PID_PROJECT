class PIDController:
    def __init__(self, Kp, Ki, Kd):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.previous_error = 0
        self.integral = 0

    def run_step(self, error, dt):
        # Calculate proportional term
        p = self.Kp * error
        
        # Calculate integral term
        self.integral += error * dt
        i = self.Ki * self.integral
        
        # Calculate derivative term
        derivative = (error - self.previous_error) / dt
        d = self.Kd * derivative
        
        # Update previous error
        self.previous_error = error
        
        return p + i + d

    def reset(self):
        self.previous_error = 0
        self.integral = 0