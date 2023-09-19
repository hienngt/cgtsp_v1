from t1 import DataUtil, Graph, CGTSP, TSP_constraint, TSP, Tour

import pandas as pd
import matplotlib as plt
# corr_df = pd.DataFrame(index=range(65), columns=range(65))
cgtsp = CGTSP()
distance_df = pd.DataFrame(index=range(cgtsp.graph.num_nodes), columns=range(cgtsp.graph.num_nodes))

# def transform_ctgtsp_to_tsp_constraint(cgtsp: CGTSP):
num_clusters = cgtsp.graph.num_clusters
num_subclusters = cgtsp.graph.num_subclusters

corr_df = cgtsp.graph.corr_df
cluster_df = cgtsp.graph.cluster_df

upper_bound = num_subclusters * max(cgtsp.graph.distance_df)
p1 = upper_bound
p2 = upper_bound * num_subclusters

# tsp_constraint
for c in range(1, num_clusters+1):
    num_subcluster_in_cluster = cluster_df.loc[c].index.get_level_values('subcluster_id').nunique()
    print(f'{num_subcluster_in_cluster=}')
    for s in range(1, num_subcluster_in_cluster+1):
        df_inside_subcluster = corr_df.query(f'cluster_id == {c} and subcluster_id == {s}').reset_index(drop=True)
        df_outside_subcluster = corr_df.query(f'cluster_id == {c} and subcluster_id != {s}').reset_index(drop=True)
        df_outside_cluster = corr_df.query(f'cluster_id != {c}').reset_index(drop=True)

        for i in range(len(df_inside_subcluster)):
            if i == len(df_inside_subcluster) - 1:
                distance_df.loc[df_inside_subcluster.loc[i]['index'], df_inside_subcluster.loc[0]['index']] = 0
            else:
                distance_df.loc[df_inside_subcluster.loc[i]['index'], df_inside_subcluster.loc[i+1]['index']] = 0

            if i == 0:
                k = len(df_inside_subcluster) - 1
            else:
                k = i - 1

            for j in range(len(df_outside_subcluster)):
                distance_df.loc[df_inside_subcluster.loc[i]['index'], df_outside_subcluster.loc[j]['index']] = \
                    cgtsp.graph.distance_df.loc[df_inside_subcluster.loc[k]['index'], df_outside_subcluster.loc[j]['index']] \
                    + p1
            for j in range(len(df_outside_cluster)):
                distance_df.loc[df_inside_subcluster.loc[i]['index'], df_outside_cluster.loc[j]['index']] = \
                    cgtsp.graph.distance_df.loc[df_inside_subcluster.loc[k]['index'], df_outside_cluster.loc[j]['index']]\
                    + p1 + p2


distance_df.fillna(0, inplace=True)


cgtsp.graph.distance_df = distance_df

tsp = TSP(graph=Graph(distance_df=distance_df, corr_df=cgtsp.graph.corr_df))