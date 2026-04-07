"""
Simple SimPy simulation: customers queue for a limited number of service clerks.
"""

import random
import statistics

import simpy


RANDOM_SEED = 42
NUM_CLERKS = 2
SIM_TIME = 120.0  # minutes
MEAN_INTERARRIVAL = 4.0  # average minutes between customer arrivals
MEAN_SERVICE = 6.0  # average minutes of service per customer


def customer(
    env: simpy.Environment,
    clerks: simpy.Resource,
    wait_times: list[float],
    service_durations: list[float],
):
    arrival = env.now
    with clerks.request() as req:
        yield req
        waited = env.now - arrival
        wait_times.append(waited)
        service = random.expovariate(1.0 / MEAN_SERVICE)
        service_durations.append(service)
        yield env.timeout(service)


def customer_generator(
    env: simpy.Environment,
    clerks: simpy.Resource,
    wait_times: list[float],
    service_durations: list[float],
):
    while True:
        interarrival = random.expovariate(1.0 / MEAN_INTERARRIVAL)
        yield env.timeout(interarrival)
        env.process(customer(env, clerks, wait_times, service_durations))


def main() -> None:
    random.seed(RANDOM_SEED)
    wait_times: list[float] = []
    service_durations: list[float] = []

    env = simpy.Environment()
    clerks = simpy.Resource(env, capacity=NUM_CLERKS)
    env.process(customer_generator(env, clerks, wait_times, service_durations))
    env.run(until=SIM_TIME)

    total_service = sum(service_durations)
    capacity_minutes = NUM_CLERKS * SIM_TIME
    utilization = total_service / capacity_minutes if capacity_minutes else 0.0

    print(f"Simulation ran for {SIM_TIME} time units.")
    print(f"Clerks: {NUM_CLERKS}")
    print(f"Customers served: {len(wait_times)}")
    if wait_times:
        print(f"Average wait in queue: {statistics.mean(wait_times):.2f}")
        print(f"Median wait: {statistics.median(wait_times):.2f}")
        print(f"Max wait: {max(wait_times):.2f}")
    print(f"Approx. clerk utilization: {utilization:.0%}")


if __name__ == "__main__":
    main()
