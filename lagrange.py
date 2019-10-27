import torch
import numpy as np

from matplotlib import pyplot as plt

class trebuchet_system():
    def q_to_cartesian(self, q):
        return q

    def cartesian_to_kinetic(self, x, x_dot):
        return x_dot[:, 0, 0] ** 2 + x_dot[:, 1, 0] ** 2

    def cartesian_to_potential(self, x, x_dot):
        # gravitational potential
        return x[:, 1, 0]

    def x0(self):
        return torch.tensor([[[-1.], [0]]])

    def xf(self):
        return torch.tensor([[[1.], [0]]])

    def new_q(self):
        return torch.rand(10, 2, 1) # timestep, (x, y), boxed value

    def render(self, x):
        plt.clf()
        plt.plot([-2, 2], [0, 0], 'g-') # "the ground"
        plt.plot(x[:, 0, 0], x[:, 1, 0], '.-')

class orbit_system():
    def q_to_cartesian(self, q):
        return q

    def cartesian_to_kinetic(self, x, x_dot):
        return x_dot[:, 0, 0] ** 2 + x_dot[:, 1, 0] ** 2

    def cartesian_to_potential(self, x, x_dot):
        # keplerian central force
        return -(x[:, :, 0].pow(2)).sum(axis=1).pow(0.5).pow(-1)

    def x0(self):
        return torch.tensor([[[-1.], [-1.]]])

    def xf(self):
        return torch.tensor([[[1.], [1.]]])

    def new_q(self):
        return torch.rand(10, 2, 1) # timestep, (x, y), boxed value

    def render(self, x):
        plt.clf()
        plt.plot(0, 0, 'yo') # the "sun"
        plt.plot(x[:, 0, 0], x[:, 1, 0], '.-')

class cart_pole_system():
    # magic numbers
    POLE_SIZE  = 10
    POLE_MASS = 5
    CART_MASS = 1.2
    GRAVITY = 0.1

    X = 0
    Y = 1
    PHI = 2
    THETA = 3

    def q_to_cartesian(self, q):
        return torch.stack([
            q[:, self.X, :],
            q[:, self.Y, :],
            q[:, self.X, :] + q[:, self.PHI, :].sin() * self.POLE_SIZE * q[:, self.THETA].cos(),
            q[:, self.Y, :] + q[:, self.PHI, :].sin() * self.POLE_SIZE * q[:, self.THETA].sin(),
            q[:, self.PHI, :].cos() * self.POLE_SIZE,
        ], 1)

    def cartesian_to_kinetic(self, x, x_dot):
        return 0.5 * self.CART_MASS * (x_dot[:, :2, 0]**2).sum(axis=1) + 0.5 * self.POLE_MASS * (x_dot[:, 2:, 0]**2).sum(axis=1)

    def cartesian_to_potential(self, x, x_dot):
        return -x[:, 4, :] * self.POLE_MASS * self.GRAVITY

    def x0(self):
        return self.q_to_cartesian(
            torch.tensor([[  [0.], [0.], [0.1], [0.]  ]])
        )

    def xf(self):
        return self.q_to_cartesian(
            torch.tensor([[  [10.], [10.], [1.2], [1.5]  ]])
        )

    def new_q(self):
        return torch.rand(20, 4, 1) # timestep, (x, y, phi, theta), boxed value

    def render(self, x):
        global i
        from mpl_toolkits.mplot3d import Axes3D
        plt.clf()
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlim(-20, 20)
        ax.set_ylim(-20, 20)
        ax.set_zlim(-5, 10)
        ax.view_init(20, i / 1000 * 3)
        for t in range(x.shape[0]):
            ax.plot(
                [x[t, 0, 0], x[t, 2, 0]],
                [x[t, 1, 0], x[t, 3, 0]],
                [0, x[t, 4, 0]]
            )

system = cart_pole_system()

q = system.new_q()
x0 = system.x0()
xf = system.xf()

i = 0
while True:
    i = i + 1
    q.requires_grad_()

    x = system.q_to_cartesian(q)

    x_augmented_initial = torch.cat([x0, x])
    x_augmented_final   = torch.cat([x, xf])
    x_dot = x_augmented_final - x_augmented_initial

    x = x_augmented_final

    T = system.cartesian_to_kinetic(x, x_dot)
    U = system.cartesian_to_potential(x, x_dot)
    L = T - U
    A = L.sum()

    A.backward()
    q = q - 0.00001 * q.grad
    q = q.detach()
    
    if i % 1000 == 1:
        x_out = torch.cat([x0, x, xf]).detach().numpy()
        system.render(x_out)
        plt.savefig('out-%010d.png' % i)
        plt.close()
