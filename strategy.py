from flwr.server.strategy import FedAvg
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays

class HybridSyncAsyncStrategy(FedAvg):
    def __init__(self, async_client_ids=None, async_weight=0.5, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.async_client_ids = set(async_client_ids or [])
        self.async_updates = []
        self.async_weight = async_weight
        self.latest_weights = None

    def aggregate_fit(self, rnd, results, failures):
        # Separate sync and async client results
        sync_results = []
        new_async_updates = []

        for client, fit_res in results:
            if client.cid in self.async_client_ids:
                # Collect async updates
                new_async_updates.append((client.cid, fit_res.parameters, fit_res.num_examples))
            else:
                # Sync updates
                sync_results.append((client, fit_res))

        # Aggregate synchronous results using FedAvg
        agg_sync = super().aggregate_fit(rnd, sync_results, failures)

        # If no sync results, return None (no aggregation possible)
        if agg_sync is None:
            return None

        # Unpack parameters and metrics returned by FedAvg
        agg_sync_parameters, agg_sync_metrics = agg_sync

        # Convert aggregated parameters to ndarrays
        agg_sync_weights = parameters_to_ndarrays(agg_sync_parameters)

        # Initialize latest weights if not set
        if self.latest_weights is None:
            self.latest_weights = agg_sync_weights

        # Add new async updates to existing async updates list
        self.async_updates.extend(new_async_updates)

        # Aggregate async updates if any
        if self.async_updates:
            total_async_examples = sum(num_ex for _, _, num_ex in self.async_updates)
            async_weights_sum = None
            for _, params, num_examples in self.async_updates:
                w = parameters_to_ndarrays(params)
                weighted_w = [layer * num_examples for layer in w]
                if async_weights_sum is None:
                    async_weights_sum = weighted_w
                else:
                    async_weights_sum = [a + b for a, b in zip(async_weights_sum, weighted_w)]
            async_avg = [layer / total_async_examples for layer in async_weights_sum]
        else:
            async_avg = None

        # Combine sync and async weights according to async_weight
        if async_avg is not None:
            combined_weights = [
                (1 - self.async_weight) * s + self.async_weight * a
                for s, a in zip(agg_sync_weights, async_avg)
            ]
        else:
            combined_weights = agg_sync_weights

        # Clear async updates after aggregation
        self.async_updates = []

        self.latest_weights = combined_weights

        combined_parameters = ndarrays_to_parameters(combined_weights)

        # Aggregate metrics from synchronous clients only (example: accuracy)
        metrics_aggregated = {}
        if sync_results:
            total_examples = sum(fit_res.num_examples for _, fit_res in sync_results)
            accuracy_sum = 0.0
            count = 0
            for _, fit_res in sync_results:
                metrics = fit_res.metrics or {}
                if "accuracy" in metrics:
                    accuracy_sum += metrics["accuracy"] * fit_res.num_examples
                    count += fit_res.num_examples
            if count > 0:
                metrics_aggregated["accuracy"] = accuracy_sum / count

        # Merge metrics from FedAvg (agg_sync_metrics) if needed
        # Example: metrics_aggregated.update(agg_sync_metrics)

        # Return parameters and metrics tuple
        return combined_parameters, metrics_aggregated
