class PID():

    def __init__(self, Kp=1.0, Ki=0.0, Kd=0.0, Ku = 1.0, sample_time=0.01, output_limits=(-1.0, 1.0), integeral_limits=(-0.5, 0.5)):

        self.integral = 0
        self.e_prev = 0
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.sample_time = sample_time
        self.output_limits = output_limits
        self.integral_limits = integeral_limits

    def update (self, setpoint, measurement):
        # PID calculations
        e = setpoint - measurement
        P = self.Kp*e
        self.integral = self.integral + self.Ki*e* self.sample_time
        #limit integeral output
        if(self.integral > self.integral_limits[1]):
            self.integral = self.integral_limits[1]
        elif(self.integral < self.integral_limits[0]):
            self.integral = self.integral_limits[0]

        D = self.Kd*(e - self.e_prev)/self.sample_time
        U = P + self.integral + D
        #limit output
        if(U < self.output_limits[0]):
            U = self.output_limits[0]
        elif(U > self.output_limits[1]):
            U = self.output_limits[1]

        # update stored data for next iteration
        self.e_prev = e
        return U
    
    def reset (self,):
        self.integral = 0
        self.e_prev = 0
        




