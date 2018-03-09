import numpy as np
import matplotlib.pyplot as plt


class EXIT_trace():
    def __init__(self, x_down, y_down, x_up, y_up):
        self.x_down = x_down
        self.y_down = y_down
        self.x_up = x_up
        self.y_up = y_up

    def _horizontal_step(self):
        a = np.argsort((self.trace_y[-1]-self.y_down)**2)
        assert a[0]-a[1]==1 or a[0]-a[1]==-1
        y0 = self.y_down[a[0]]
        y1 = self.y_down[a[1]]
        x0 = self.x_down[a[0]]
        x1 = self.x_down[a[1]]
        new_x = (self.trace_y[-1]-y1)/(y0-y1)*(x0-x1)+x1
        self.trace_x.append(new_x)
        self.trace_y.append(self.trace_y[-1])

    def _vertical_step(self):
        a = np.argsort((self.trace_x[-1]-self.x_up)**2)
        assert a[0]-a[1]==1 or a[0]-a[1]==-1
        y0 = self.y_up[a[0]]
        y1 = self.y_up[a[1]]
        x0 = self.x_up[a[0]]
        x1 = self.x_up[a[1]]
        new_y = (self.trace_x[-1]-x1)/(x0-x1)*(y0-y1)+y1
        self.trace_x.append(self.trace_x[-1])
        self.trace_y.append(new_y)

    def trace(self, max_iterations=200):
        self.trace_x = [self.x_up[0]]
        self.trace_y = [self.y_up[0]]
        for i in range(max_iterations):
            self._horizontal_step()
            self._vertical_step()
            a = self.trace_y[-1]
            b = self.trace_y[-2]
            err_y = 2*(a-b)/(a+b)
            a = self.trace_x[-2]
            b = self.trace_x[-3]
            err_x = 2*(a-b)/(a+b)
            err = 0.5*(err_x**2+err_y**2)
            if err < 1e-10 or self.trace_x[-1]>=1. or self.trace_y[-1]>=1.:
                break

    def plot(self):
        plt.plot(self.x_down, self.y_down)
        plt.plot(self.x_up, self.y_up)
        plt.plot(self.trace_x, self.trace_y)
        N_iter = len(self.trace_x)
        x = self.trace_x[-1]
        y = self.trace_y[-1]
        str = 'N_iter={}, stuck at {:.3f},{:.3f}'.format(N_iter, x, y)
        plt.title(str)
        plt.grid()
        plt.show()


if __name__ == '__main__':
    x_down = np.linspace(0, 1, 11)
    x_up = x_down
    y_down = np.linspace(0, 1, 11)
    y_up = np.linspace(.3, 1, 11)
    exit_trace = EXIT_trace(x_down, y_down, x_up, y_up)
    exit_trace.trace()
    exit_trace.plot()
    
    
