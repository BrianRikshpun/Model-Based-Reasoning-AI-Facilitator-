"""
Simple SimPy simulation: customers queue for a limited number of service clerks.
"""

import statistics

from queue_simulation import run_simulation

RANDOM_SEED = 42
NUM_CLERKS = 2
SIM_TIME = 120.0  # minutes
MEAN_INTERARRIVAL = 4.0  # average minutes between customer arrivals
MEAN_SERVICE = 6.0  # average minutes of service per customer


def main() -> None:
    r = run_simulation(
        RANDOM_SEED,
        NUM_CLERKS,
        SIM_TIME,
        MEAN_INTERARRIVAL,
        MEAN_SERVICE,
    )
    utilization = r["utilization"]

    print(f"Simulation ran for {SIM_TIME} time units.")
    print(f"Clerks: {NUM_CLERKS}")
    print(f"Customers served: {r['customers_started']}")
    if r["mean_wait"] is not None:
        print(f"Average wait in queue: {r['mean_wait']:.2f}")
        print(f"Median wait: {r['median_wait']:.2f}")
        print(f"Max wait: {r['max_wait']:.2f}")
    print(f"Approx. clerk utilization: {utilization:.0%}")


if __name__ == "__main__":
    main()
