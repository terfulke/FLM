from flwr.server.strategy import FedAvg
import flwr as fl
from flwr.common import parameters_to_ndarrays, ndarrays_to_parameters
from collections import defaultdict


class FedSAStrategy(FedAvg):
    def __init__(self, M=3, tau_0=3, base_lr=1.0, num_total_clients=10, evaluate_metrics_aggregation_fn=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.M = M
        self.tau_0 = tau_0
        self.round = 0
        self.base_lr = base_lr
        self.num_total_clients = num_total_clients
        self.latest_weights = None
        self.client_staleness = defaultdict(lambda: 0)
        self.participation_count = defaultdict(lambda: 0)
        self.evaluate_metrics_aggregation_fn = evaluate_metrics_aggregation_fn
        self.last_configured_cids = []

    def aggregate_fit(self, rnd, results, failures):
        results = [r for r in results if r[1].num_examples > 0]
        self.round = rnd

        # Initialize staleness for any new clients
        for client, _ in results:
            cid = client.cid
            if cid not in self.client_staleness:
                self.client_staleness[cid] = 0

        # Filter out clients whose staleness exceeds tau_0
        filtered_results = []
        for client, fit_res in results:
            cid = client.cid
            if self.client_staleness[cid] <= self.tau_0:
                filtered_results.append((client, fit_res))
            else:
                print(f"[FedSA] Skipping stale client {cid} with staleness {self.client_staleness[cid]}")

        if len(filtered_results) < self.M:
            print(f"[FedSA] Round {rnd}: Only {len(filtered_results)} usable updates received, waiting for at least {self.M}")
            return ndarrays_to_parameters(self.latest_weights) if self.latest_weights else None, {}

        # Select first M non-stale clients
        selected = filtered_results[:self.M]
        self.last_configured_cids = [client.cid for client, _ in selected]

        total_examples = sum(fit_res.num_examples for _, fit_res in selected)
        weighted_sum = None

        # Aggregate weights
        for client, fit_res in selected:
            cid = client.cid
            self.participation_count[cid] += 1
            self.client_staleness[cid] = 0  # Reset staleness for used client

            weights = parameters_to_ndarrays(fit_res.parameters)
            weighted = [layer * fit_res.num_examples for layer in weights]

            if weighted_sum is None:
                weighted_sum = weighted
            else:
                weighted_sum = [a + b for a, b in zip(weighted_sum, weighted)]

        # Update staleness for all known clients that were NOT selected
        used_cids = set(self.last_configured_cids)
        for cid in self.client_staleness:
            if cid not in used_cids:
                self.client_staleness[cid] += 1

        # Log staleness info
        print(f"[FedSA] Client staleness map: {dict(self.client_staleness)}")

        # Final aggregation
        agg_weights = [layer / total_examples for layer in weighted_sum]
        self.latest_weights = agg_weights
        agg_parameters = ndarrays_to_parameters(agg_weights)

        # Aggregate metrics if function provided
        metrics_list = [(fit_res.num_examples, fit_res.metrics or {}) for _, fit_res in selected]
        aggregated_metrics = (
            self.evaluate_metrics_aggregation_fn(metrics_list)
            if self.evaluate_metrics_aggregation_fn
            else {}
        )

        return agg_parameters, aggregated_metrics

    def configure_fit(self, server_round, parameters, client_manager):
        # Override to inject adaptive learning rates
        selected_clients = list(client_manager.sample(num_clients=self.M, min_num_clients=self.M))
        self.last_configured_cids = [client.cid for client in selected_clients]

        instructions = []

        for client in selected_clients:
            cid = client.cid
            f_i = max(1, self.participation_count[cid])
            eta_i = self.base_lr / (self.num_total_clients * f_i)

            config = {
                "lr": eta_i
            }

            instructions.append((client, fl.server.client_proxy.FitIns(parameters, config)))

        return instructions
