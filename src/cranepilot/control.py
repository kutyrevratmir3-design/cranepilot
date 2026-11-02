"""Simulation-only MPC scaffold with explicit model/plant separation.

State feedback is assumed in this first control experiment. It is not a
hardware controller experiment or validation of the neural forecasting model.
"""
from dataclasses import dataclass
import time
import numpy as np
from scipy.linalg import expm, solve_discrete_are
from scipy.optimize import minimize
from .physics import G


@dataclass
class PlantParameters:
    length: float
    actuator_tau: float = .10
    damping: float = .03
    deadzone: float = .02
    acceleration_gain: float = 1.0


def plant_rhs(state, command, p: PlantParameters, disturbance=0.0):
    x, v, theta, rate, acceleration = np.asarray(state)
    effective_command = np.sign(command) * max(abs(command)-p.deadzone, 0.0) * p.acceleration_gain
    return np.array([v, acceleration,
                     rate, -(G*np.sin(theta)+acceleration*np.cos(theta))/p.length-p.damping*rate+disturbance,
                     (effective_command-acceleration)/p.actuator_tau])


def rk4_step(state, command, p, dt=.05, disturbance=0.0, substeps=5):
    out = np.asarray(state, dtype=float).copy()
    h = dt/substeps
    for _ in range(substeps):
        k1 = plant_rhs(out,command,p,disturbance)
        k2 = plant_rhs(out+h*k1/2,command,p,disturbance)
        k3 = plant_rhs(out+h*k2/2,command,p,disturbance)
        k4 = plant_rhs(out+h*k3,command,p,disturbance)
        out += h*(k1+2*k2+2*k3+k4)/6
    return out


class LinearMPC:
    """Bounded-input quadratic MPC; no claim of hard state-constraint guarantees."""
    def __init__(self, length, dt=.05, horizon=40, max_acceleration=.8,
                 actuator_tau=.10, damping=.03):
        self.length, self.dt, self.horizon = length, dt, horizon
        self.bound = max_acceleration
        ac = np.zeros((5,5))
        ac[0,1] = ac[2,3] = 1
        ac[1,4] = 1
        ac[3,2], ac[3,3], ac[3,4] = -G/length, -damping, -1/length
        ac[4,4] = -1/actuator_tau
        bc = np.zeros((5,1)); bc[4,0] = 1/actuator_tau
        block = np.zeros((6,6)); block[:5,:5], block[:5,5:] = ac, bc
        discrete = expm(block*dt)
        a,b = discrete[:5,:5], discrete[:5,5:]
        self.q = np.diag([20., 2., 25., 1., .02])
        r = np.array([[.15]])
        terminal = solve_discrete_are(a,b,self.q,r)
        self.feedback = np.linalg.solve(r+b.T@terminal@b, b.T@terminal@a)[0]
        f = np.vstack([np.linalg.matrix_power(a,j+1) for j in range(horizon)])
        g = np.zeros((5*horizon,horizon))
        for j in range(horizon):
            for k in range(j+1):
                g[j*5:(j+1)*5,k] = (np.linalg.matrix_power(a,j-k)@b)[:,0]
        qbar = np.kron(np.eye(horizon),self.q)
        qbar[-5:,-5:] = terminal
        # Penalise changes inside the planned command sequence.
        delta = np.eye(horizon)-np.eye(horizon,k=-1)
        delta[0,0] = 0
        self.hessian = g.T@qbar@g + .15*np.eye(horizon) + .1*delta.T@delta
        self.linear = g.T@qbar@f
        self.warm = np.zeros(horizon)
        self.failures = 0
        self.times = []

    def __call__(self,state,target):
        error = np.asarray(state)-np.array([target,0,0,0,0])
        linear = self.linear@error
        started = time.perf_counter()
        result = minimize(lambda u: float(u@self.hessian@u+2*linear@u), self.warm,
                          jac=lambda u: 2*(self.hessian@u+linear), method="L-BFGS-B",
                          bounds=[(-self.bound,self.bound)]*self.horizon,
                          options={"maxiter":50,"ftol":1e-8,"gtol":1e-5,"maxls":10})
        self.times.append(time.perf_counter()-started)
        if result.success and np.isfinite(result.x).all():
            command = result.x[0]
            self.warm = np.r_[result.x[1:],result.x[-1]]
        else:
            self.failures += 1
            command = np.clip(-self.feedback@error,-self.bound,self.bound)
            self.warm.fill(command)
        return float(command)


def simulate_episode(controller,p,target=.4,duration=12.,dt=.05,seed=0):
    rng = np.random.default_rng(seed)
    grid = np.arange(int(round(duration/dt))+1)*dt
    # One disturbance per physical control interval, shared across controllers.
    disturbances = rng.normal(0,.025,len(grid))
    state = np.array([0.,0.,rng.uniform(-.05,.05),0.,0.])
    states,commands = [state.copy()], []
    for j,t in enumerate(grid[:-1]):
        command = controller(state,target)
        state = rk4_step(state,command,p,dt,disturbances[j])
        if not np.isfinite(state).all():
            raise RuntimeError("Simulation diverged; must retain failed episode")
        states.append(state.copy()); commands.append(command)
    states,commands = np.array(states),np.array(commands)
    theta_deg = np.rad2deg(states[:,2])
    settled = (np.abs(states[:,0]-target)<.01)&(np.abs(states[:,1])<.02)&(np.abs(theta_deg)<1.)
    remaining_ok = np.logical_and.accumulate(settled[::-1])[::-1]
    settling = float(grid[np.flatnonzero(remaining_ok)[0]]) if remaining_ok.any() else None
    metrics = {"residual_rms_deg":float(np.sqrt(np.mean(theta_deg[grid>=duration-2]**2))),
               "peak_angle_deg":float(np.max(np.abs(theta_deg))),
               "sway_ise_deg2_s":float(np.trapezoid(theta_deg**2,grid)),
               "final_position_error_m":float(abs(states[-1,0]-target)),
               "settling_time_s":settling, "settled":settling is not None,
               "effort_proxy":float(np.sum(commands**2)*dt),
               "angle_over_15deg":bool(np.any(np.abs(theta_deg)>15)),
               "speed_over_0p5ms":bool(np.any(np.abs(states[:,1])>.5)),
               "solver_failures":controller.failures,
               "solve_p99_ms":float(np.quantile(controller.times,.99)*1000),
               "solve_max_ms":float(np.max(controller.times)*1000),
               "deadline_misses":int(np.sum(np.array(controller.times)>dt))}
    return metrics,{"time":grid,"states":states,"commands":commands,"disturbance":disturbances}

