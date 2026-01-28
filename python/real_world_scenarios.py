"""Real-world scenario generators and benchmarks for almo-mcf-rs.

These scenarios generate NetworkX graphs that model large-scale, real-world
min-cost flow problems. They are intended to stress performance and highlight
the benefits of almo-mcf-rs relative to traditional NetworkX solvers.
"""
from __future__ import annotations

import random
import time
from dataclasses import dataclass
from typing import Iterable

import networkx as nx

from almo_mcf import min_cost_flow as almo_min_cost_flow
from almo_mcf import min_cost_flow_cost as almo_min_cost_flow_cost


@dataclass(frozen=True)
class Scenario:
    name: str
    graph: nx.DiGraph
    description: str


def assign_balanced_demands(
    G: nx.DiGraph,
    nodes: Iterable,
    total_flow: int,
    num_sources: int,
    num_sinks: int,
    rng: random.Random,
) -> tuple[list, list]:
    """Assign balanced supplies/demands across random sources and sinks."""
    node_list = list(nodes)
    if num_sources + num_sinks > len(node_list):
        raise ValueError("Number of sources and sinks exceeds node count.")

    sources = rng.sample(node_list, num_sources)
    remaining = [node for node in node_list if node not in sources]
    sinks = rng.sample(remaining, num_sinks)

    for node in node_list:
        G.nodes[node]["demand"] = 0

    supply_per_source, supply_remainder = divmod(total_flow, num_sources)
    demand_per_sink, demand_remainder = divmod(total_flow, num_sinks)

    for source in sources:
        G.nodes[source]["demand"] = -supply_per_source
    for idx in range(supply_remainder):
        G.nodes[sources[idx]]["demand"] -= 1

    for sink in sinks:
        G.nodes[sink]["demand"] = demand_per_sink
    for idx in range(demand_remainder):
        G.nodes[sinks[idx]]["demand"] += 1

    return sources, sinks


def add_edge_attributes(
    G: nx.DiGraph,
    rng: random.Random,
    min_cap: int = 1,
    max_cap: int = 100,
    min_cost: int = 1,
    max_cost: int = 10,
) -> None:
    for u, v in G.edges():
        G.edges[u, v]["capacity"] = rng.randint(min_cap, max_cap)
        G.edges[u, v]["weight"] = rng.randint(min_cost, max_cost)


def build_airline_crew_scheduling_graph(
    rng: random.Random,
    num_airports: int = 50,
    num_flights: int = 500,
    num_crews: int = 200,
    max_connections: int = 3,
) -> Scenario:
    """Time-expanded flight connection graph for crew scheduling."""
    G = nx.DiGraph()
    airports = [f"APT_{idx}" for idx in range(num_airports)]
    flight_nodes = []
    flights = []

    for idx in range(num_flights):
        origin, dest = rng.sample(airports, 2)
        depart = rng.randint(0, 1440)
        arrive = depart + rng.randint(30, 480)
        flight_id = f"FL_{idx}"
        flights.append((flight_id, origin, dest, depart, arrive))
        flight_nodes.append(flight_id)
        G.add_node(flight_id, demand=0)

    origin_airports = sorted({origin for _fid, origin, _dest, _d, _a in flights})
    dest_airports = sorted({dest for _fid, _origin, dest, _d, _a in flights})
    crew_pools = [f"CrewPool_{apt}" for apt in origin_airports]
    sinks = [f"CrewSink_{apt}" for apt in dest_airports]
    G.add_nodes_from(crew_pools, demand=0)
    G.add_nodes_from(sinks, demand=0)

    crew_supply = min(num_crews, len(flights))
    sampled_flights = rng.sample(flights, crew_supply)
    origin_counts = {pool: 0 for pool in crew_pools}
    dest_counts = {sink: 0 for sink in sinks}
    for _flight_id, origin, dest, _depart, _arrive in sampled_flights:
        origin_counts[f"CrewPool_{origin}"] += 1
        dest_counts[f"CrewSink_{dest}"] += 1

    for pool, count in origin_counts.items():
        G.nodes[pool]["demand"] = -count
    for sink, count in dest_counts.items():
        G.nodes[sink]["demand"] = count

    for flight_id, origin, dest, _depart, _arrive in flights:
        pool_node = f"CrewPool_{origin}"
        G.add_edge(pool_node, flight_id)
        G.add_edge(flight_id, f"CrewSink_{dest}")

    flights_by_origin = {apt: [] for apt in airports}
    for flight_id, origin, dest, depart, arrive in flights:
        flights_by_origin[origin].append((flight_id, dest, depart, arrive))

    for flight_id, _origin, dest, _depart, arrive in flights:
        next_flights = [
            f
            for f in flights_by_origin[dest]
            if f[2] >= arrive + 30
        ]
        rng.shuffle(next_flights)
        for next_flight_id, _n_dest, _n_depart, _n_arrive in next_flights[:max_connections]:
            G.add_edge(flight_id, next_flight_id)

    add_edge_attributes(G, rng, min_cap=1, max_cap=1, min_cost=1, max_cost=100)
    for pool in crew_pools:
        for u, v in G.edges(pool):
            G.edges[u, v]["weight"] = 0
    for sink in sinks:
        for u, v in G.in_edges(sink):
            G.edges[u, v]["weight"] = 0

    return Scenario(
        name="Airline Crew Scheduling",
        graph=G,
        description=(
            "Assign crews to flight legs with layover-compatible connections."
        ),
    )


def build_financial_arbitrage_graph(
    rng: random.Random,
    num_assets: int = 1000,
    num_edges: int = 10000,
    total_volume: int = 10_000,
) -> Scenario:
    """Dense asset exchange graph with negative costs for profitable trades."""
    G = nx.gnm_random_graph(num_assets, num_edges, directed=True, seed=rng)
    G = nx.DiGraph(G)
    assets = list(G.nodes())

    for node in assets:
        G.nodes[node]["demand"] = 0

    super_source = "SuperSource"
    super_sink = "SuperSink"
    G.add_node(super_source, demand=-total_volume)
    G.add_node(super_sink, demand=total_volume)

    for asset in assets:
        G.add_edge(super_source, asset, capacity=total_volume, weight=0)
        G.add_edge(asset, super_sink, capacity=total_volume, weight=0)

    for u, v in G.edges():
        if u in (super_source, super_sink) or v in (super_source, super_sink):
            continue
        G.edges[u, v]["capacity"] = rng.randint(1_000, 1_000_000)
        G.edges[u, v]["weight"] = rng.randint(-10, 10)

    return Scenario(
        name="Financial Arbitrage Detection",
        graph=G,
        description=(
            "Asset exchange network with negative-cost edges to represent profit."
        ),
    )


def build_warehouse_slotting_graph(
    rng: random.Random,
    num_products: int = 2000,
    num_slots: int = 2000,
    density: float = 0.05,
) -> Scenario:
    """Bipartite graph for warehouse slotting optimization."""
    G = nx.DiGraph()
    products = [f"Product_{idx}" for idx in range(num_products)]
    slots = [f"Slot_{idx}" for idx in range(num_slots)]
    G.add_nodes_from(products, demand=0)
    G.add_nodes_from(slots, demand=0)

    product_demands = [rng.randint(1, 10) for _ in products]
    total_flow = sum(product_demands)
    slot_capacities = [rng.randint(1, 10) for _ in slots]
    capacity_scale = total_flow / max(1, sum(slot_capacities))
    slot_capacities = [max(1, int(capacity_scale * cap)) for cap in slot_capacities]

    for product, demand in zip(products, product_demands):
        G.nodes[product]["demand"] = -demand
    for slot, capacity in zip(slots, slot_capacities):
        G.nodes[slot]["demand"] = capacity

    total_supply = -sum(G.nodes[p]["demand"] for p in products)
    total_demand = sum(G.nodes[s]["demand"] for s in slots)
    if total_supply != total_demand:
        adjust = total_supply - total_demand
        G.nodes[slots[0]]["demand"] += adjust

    for product in products:
        for slot in slots:
            if rng.random() < density:
                G.add_edge(product, slot)

    add_edge_attributes(G, rng, min_cap=1, max_cap=10, min_cost=1, max_cost=500)
    return Scenario(
        name="Warehouse Slotting Optimization",
        graph=G,
        description=(
            "Assign products to slots with costs driven by travel distance."
        ),
    )


def build_telecom_routing_graph(
    rng: random.Random,
    num_nodes: int = 5000,
    attachment: int = 3,
    total_flow: int = 50_000,
    num_sources: int = 50,
    num_sinks: int = 200,
) -> Scenario:
    """Scale-free telecom routing graph with multiple sources/sinks."""
    base_graph = nx.barabasi_albert_graph(num_nodes, attachment, seed=rng)
    G = nx.DiGraph()
    for u, v in base_graph.edges():
        G.add_edge(u, v)
        G.add_edge(v, u)

    assign_balanced_demands(
        G,
        G.nodes(),
        total_flow=total_flow,
        num_sources=num_sources,
        num_sinks=num_sinks,
        rng=rng,
    )
    add_edge_attributes(G, rng, min_cap=10, max_cap=1_000, min_cost=1, max_cost=50)

    return Scenario(
        name="Telecommunications Network Routing",
        graph=G,
        description="Route bandwidth across a scale-free ISP-like topology.",
    )


def benchmark_graph(
    scenario: Scenario,
    run_networkx: bool = True,
) -> None:
    """Benchmark almo-mcf-rs vs NetworkX on a scenario graph."""
    G = scenario.graph
    print(
        f"{scenario.name}: nodes={G.number_of_nodes()} edges={G.number_of_edges()}"
    )
    print(f"  {scenario.description}")

    start = time.perf_counter()
    flow = almo_min_cost_flow(G)
    cost = almo_min_cost_flow_cost(G, flow)
    almo_time = time.perf_counter() - start
    print(f"  almo-mcf-rs cost={cost} time={almo_time:.3f}s")

    if run_networkx:
        start = time.perf_counter()
        nx_cost = nx.min_cost_flow_cost(G)
        nx_time = time.perf_counter() - start
        print(f"  NetworkX cost={nx_cost} time={nx_time:.3f}s")
        if almo_time > 0:
            print(f"  speedup (nx/almo)={nx_time / almo_time:.2f}x")


def generate_scenarios(seed: int = 42) -> list[Scenario]:
    rng = random.Random(seed)
    return [
        build_airline_crew_scheduling_graph(rng),
        build_financial_arbitrage_graph(rng),
        build_warehouse_slotting_graph(rng),
        build_telecom_routing_graph(rng),
    ]


if __name__ == "__main__":
    scenarios = generate_scenarios()
    for scenario in scenarios:
        benchmark_graph(scenario, run_networkx=True)
