"""
Parameterized queue simulation (same logic as main.py).
"""

from __future__ import annotations

import random
import statistics
from typing import Any

import simpy


def run_simulation(
    random_seed: int,
    num_clerks: int,
    sim_time: float,
    mean_interarrival: float,
    mean_service: float,
) -> dict[str, Any]:
    """
    Run one replication. Returns metrics aligned with main.py bookkeeping.
    """
    random.seed(random_seed)
    wait_times: list[float] = []
    service_durations: list[float] = []

    def customer(
        env: simpy.Environment,
        clerks: simpy.Resource,
        wt: list[float],
        sd: list[float],
        svc_mean: float,
    ):
        arrival = env.now
        with clerks.request() as req:
            yield req
            wt.append(env.now - arrival)
            service = random.expovariate(1.0 / svc_mean)
            sd.append(service)
            yield env.timeout(service)

    def customer_generator(
        env: simpy.Environment,
        clerks: simpy.Resource,
        wt: list[float],
        sd: list[float],
        ia_mean: float,
        svc_mean: float,
    ):
        while True:
            interarrival = random.expovariate(1.0 / ia_mean)
            yield env.timeout(interarrival)
            env.process(customer(env, clerks, wt, sd, svc_mean))

    env = simpy.Environment()
    clerks = simpy.Resource(env, capacity=num_clerks)
    env.process(
        customer_generator(
            env, clerks, wait_times, service_durations, mean_interarrival, mean_service
        )
    )
    env.run(until=sim_time)

    total_service = sum(service_durations)
    capacity = num_clerks * sim_time
    utilization = total_service / capacity if capacity else 0.0

    return {
        "customers_started": len(wait_times),
        "mean_wait": statistics.mean(wait_times) if wait_times else None,
        "median_wait": statistics.median(wait_times) if wait_times else None,
        "max_wait": max(wait_times) if wait_times else None,
        "utilization": utilization,
        "total_planned_service": total_service,
    }


def metric_value(results: dict[str, Any], metric: str) -> float | None:
    """Return the scalar used for grading for a named metric."""
    if metric == "customers_started":
        return float(results["customers_started"])
    if metric == "mean_wait":
        v = results["mean_wait"]
        return float(v) if v is not None else None
    if metric == "median_wait":
        v = results["median_wait"]
        return float(v) if v is not None else None
    if metric == "max_wait":
        v = results["max_wait"]
        return float(v) if v is not None else None
    if metric == "utilization":
        return float(results["utilization"])
    raise ValueError(f"Unknown metric: {metric}")
