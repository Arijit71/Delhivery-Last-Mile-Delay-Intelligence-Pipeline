
from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


# Project folders.
PROJECT_FOLDER = Path(__file__).resolve().parent
DATA_FOLDER = PROJECT_FOLDER / "data"
OUTPUT_FOLDER = PROJECT_FOLDER / "outputs"


def main():

    OUTPUT_FOLDER.mkdir(exist_ok=True)

    # --------------------------------------------------
    # STEP 1: CREATE HUB PROFILE
    # --------------------------------------------------

    data = pd.read_csv(DATA_FOLDER / "modelling_table.csv")

    hubs = pd.read_csv(DATA_FOLDER / "hubs.csv")

    # Calculate the statistics for each hub.
    hub_profile = data.groupby("hub_id", sort=True).agg(

        shipments=("shipment_id", "count"),

        mean_delivery_hours=("delivery_hours", "mean"),

        delay_rate=("is_delayed", "mean"),

        mean_distance_km=("distance_km", "mean"),

        mean_n_stops=("n_stops", "mean")

    ).reset_index()

    # Keep hubs in ascending order.
    hub_profile = hub_profile.sort_values("hub_id").reset_index(drop=True)

    # Save hub profiles.
    hub_profile.to_csv(
        OUTPUT_FOLDER / "hub_profile.csv",
        index=False,
        float_format="%.4f"
    )

    # --------------------------------------------------
    # STEP 2: SCALE HUB PROFILE DATA
    # --------------------------------------------------

    profile_features = [

        "mean_delivery_hours",

        "delay_rate",

        "mean_distance_km",

        "mean_n_stops"

    ]

    scaler = StandardScaler()

    scaled_profiles = scaler.fit_transform(
        hub_profile[profile_features]
    )

    # --------------------------------------------------
    # STEP 3: K-MEANS SEPARATION EXPERIMENT
    # --------------------------------------------------

    separation_results = []

    labels_for_three = None

    for k in [2, 3, 4, 5]:

        kmeans = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=10
        )

        labels = kmeans.fit_predict(scaled_profiles)

        # Keep K=3 labels for the final segmentation.
        if k == 3:
            labels_for_three = labels

        within_distances = []

        across_distances = []

        # Calculate distances between all unordered hub pairs.
        for i, j in combinations(range(len(labels)), 2):

            distance = np.linalg.norm(
                scaled_profiles[i] - scaled_profiles[j]
            )

            if labels[i] == labels[j]:

                within_distances.append(distance)

            else:

                across_distances.append(distance)

        # Mean distance across clusters / mean distance
        # within clusters.
        ratio = (
            np.mean(across_distances) /
            np.mean(within_distances)
        )

        # Sort cluster sizes from largest to smallest.
        cluster_sizes = sorted(
            np.bincount(labels).tolist(),
            reverse=True
        )

        separation_results.append({

            "K": k,

            "across_over_within": round(float(ratio), 4),

            "cluster_sizes": "-".join(map(str, cluster_sizes))

        })

    # Save separation results.
    separation_table = pd.DataFrame(
        separation_results,
        columns=["K", "across_over_within", "cluster_sizes"]
    )

    separation_table.to_csv(
        OUTPUT_FOLDER / "part_d_separation.csv",
        index=False
    )

    # --------------------------------------------------
    # STEP 4: CREATE THREE OPERATING SEGMENTS
    # --------------------------------------------------

    # Use the K=3 clustering.
    hub_profile["cluster"] = labels_for_three

    # Calculate the mean delay rate of each cluster.
    cluster_delay_rates = hub_profile.groupby(
        "cluster"
    )["delay_rate"].mean()

    # Arrange clusters from lowest to highest delay rate.
    ordered_clusters = cluster_delay_rates.sort_values().index.tolist()

    # Assign segment names in that order.
    segment_names = [
        "Standard",
        "Watch",
        "Escalate"
    ]

    cluster_to_segment = dict(
        zip(ordered_clusters, segment_names)
    )

    hub_profile["segment"] = hub_profile["cluster"].map(
        cluster_to_segment
    )

    # Attach hub city and tier information.
    hub_segments = hub_profile.merge(
        hubs[["hub_id", "hub_city", "hub_tier"]],
        on="hub_id",
        how="left"
    )

    # Select the required columns.
    hub_segments = hub_segments[
        ["hub_id", "segment", "hub_city", "hub_tier"]
    ]

    hub_segments = hub_segments.sort_values(
        "hub_id"
    ).reset_index(drop=True)

    hub_segments.to_csv(
        OUTPUT_FOLDER / "hub_segments.csv",
        index=False
    )

    # --------------------------------------------------
    # STEP 5: CREATE SEGMENT PROFILE
    # --------------------------------------------------

    # Calculate means using hub-level profile values.
    segment_profile = hub_profile.groupby("segment").agg(

        hubs=("hub_id", "count"),

        mean_delivery_hours=("mean_delivery_hours", "mean"),

        delay_rate=("delay_rate", "mean"),

        mean_distance_km=("mean_distance_km", "mean"),

        mean_n_stops=("mean_n_stops", "mean")

    ).reset_index()

    # Keep the order Standard, Watch, Escalate.
    segment_order = {
        "Standard": 0,
        "Watch": 1,
        "Escalate": 2
    }

    segment_profile["order"] = segment_profile["segment"].map(
        segment_order
    )

    segment_profile = segment_profile.sort_values("order")

    segment_profile = segment_profile.drop(columns=["order"])

    segment_profile.to_csv(
        OUTPUT_FOLDER / "segment_profile.csv",
        index=False,
        float_format="%.4f"
    )

    # --------------------------------------------------
    # STEP 6: FIND TIER-SEGMENT MISMATCHES
    # --------------------------------------------------

    # Expected segment based on the existing hub tier.
    tier_segment = {

        1: "Standard",

        2: "Watch",

        3: "Escalate"

    }

    expected_segment = hub_segments["hub_tier"].map(
        tier_segment
    )

    # Select hubs whose actual segment differs.
    mismatches = hub_segments[
        hub_segments["segment"] != expected_segment
    ]

    mismatches.to_csv(
        OUTPUT_FOLDER / "tier_vs_segment_mismatch.csv",
        index=False
    )

    print("Part D segmentation completed successfully!")

    print("Hub profiles:", len(hub_profile))

    print("Segment sizes:", segment_profile["hubs"].tolist())


if __name__ == "__main__":
    main()
