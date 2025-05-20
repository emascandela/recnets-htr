import numpy as np
import tabulate

def is_pareto_efficient(costs, return_mask: bool = True):
    is_efficient = np.arange(costs.shape[0])
    n_points = costs.shape[0]
    next_point_index = 0  # Next index in the is_efficient array to search for
    while next_point_index<len(costs):
        nondominated_point_mask = np.any(costs<costs[next_point_index], axis=1)
        nondominated_point_mask[next_point_index] = True
        is_efficient = is_efficient[nondominated_point_mask]  # Remove dominated points
        costs = costs[nondominated_point_mask]
        next_point_index = np.sum(nondominated_point_mask[:next_point_index])+1
    if return_mask:
        is_efficient_mask = np.zeros(n_points, dtype = bool)
        is_efficient_mask[is_efficient] = True
        return is_efficient_mask
    else:
        return is_efficient


# def get_pareto()

if __name__ == '__main__':
    # with open("pareto_summary.md", 'w') as pareto_summary:

    tables = {}
    headers = ["Name", "Params", "CER", "CER Q8", "CER Q1.58", "Type"]
    type_names = [
        "Cluster",
        "Recursion",
        "Base"
    ]

    max_depth = 5

    for name in ["washington", "saint_gall", "parzival"]:

        raw_data = np.loadtxt(f"{name}2.dat", dtype=object)
        data = raw_data[:, :2].astype(np.float32)
        sorted_idx = np.argsort(data[:, 0])

        all_data = data[sorted_idx]
        all_types = raw_data[sorted_idx, 2].astype(np.int32)
        all_names = raw_data[sorted_idx, 3]
        all_q8 = raw_data[sorted_idx, 4]
        all_q1 = raw_data[sorted_idx, 5]

        tables[name] = []

        missing = []

        for pareto_depth in range(max_depth):
            mask = is_pareto_efficient(all_data)

            data = all_data[mask]
            names = all_names[mask]
            types = all_types[mask]
            q8 = all_q8[mask]
            q1 = all_q1[mask]

            sorted_idx = np.argsort(data[:, 0])

            out_data = data[sorted_idx]
            out_data = np.concatenate([[[out_data[0, 0], 10.0]], out_data, [[6e6, out_data[-1, 1]]]], axis=0)

            if pareto_depth == 0:
                np.savetxt(f"pareto_{name}2.dat", out_data, fmt="%1.6f")

            print(q8)
            missing.extend([n for (n, cer) in zip(names, q8) if (cer=="-1" or float(cer) > 80)])
            tables[name].append(tabulate.tabulate([(n, *d, q, qb, type_names[t]) for d, q, qb, n, t in zip(data, q8, q1, names, types)], headers=headers, tablefmt="github"))
            # tables[name][-1]

            all_data = all_data[~mask]
            all_names = all_names[~mask]
            all_types = all_types[~mask]
            all_q8 = all_q8[~mask]
            all_q1 = all_q1[~mask]
        
        with open(f"missing_{name}.txt", "w") as f:
            f.write("\n".join(missing))


    
    for i in range(max_depth):
        with open(f"pareto_summary_d={i+1}.md", "w") as f:
            for name, ts in tables.items():
                f.write(f"## {name}\n")
                f.write(ts[i] + '\n\n')
        