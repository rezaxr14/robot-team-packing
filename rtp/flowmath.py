"""Flow-matching math (rectified-flow / linear interpolant). Works on numpy arrays and torch tensors
(only arithmetic is used), so it is unit-tested here without torch.

x_t = (1 - t) x0 + t x1,  x0 ~ N(0, I),  x1 = expert action chunk,  target velocity v* = x1 - x0.
Sampling: Euler steps of dx/dt = v(x, t | cond) from t=0 to 1.
"""


def interpolate(x0, x1, t):
    return (1.0 - t) * x0 + t * x1


def target_velocity(x0, x1):
    return x1 - x0


def euler_integrate(velocity_fn, x0, steps: int = 8):
    x, dt = x0, 1.0 / steps
    for k in range(steps):
        x = x + dt * velocity_fn(x, k * dt)
    return x
