# Discrete waterwheel model

The simulation uses `theta`, `omega`, and one mass per bucket. Bucket angles
in the laboratory frame are

```text
alpha_i = theta + 2*pi*i/N
```

The wheel torque is

```text
I * d omega / dt = g*R * sum_i(m_i*sin(alpha_i)) - c*omega
```

Each bucket loses water exponentially. Inflow is distributed smoothly near
the top of the wheel with a von Mises profile:

```text
q_i = Q * exp(kappa*cos(alpha_i)) / sum_j exp(kappa*cos(alpha_j))
dm_i/dt = q_i - leak_rate*m_i
```

This removes the artificial discontinuity of assigning all inflow to whichever
bucket happens to be highest. The physical observables used in the plots are

```text
C1 = sum_i(m_i*cos(alpha_i))
S1 = sum_i(m_i*sin(alpha_i))
```

The attractor plot is generated exclusively from the simulated bucket
trajectory `(theta, omega, m_i)`. For the default parameters, the numerical
trajectory has a positive finite-time Lyapunov estimate and crosses
`omega=0`; these are empirical properties of this implementation, not a claim
of universal waterwheel behaviour.
