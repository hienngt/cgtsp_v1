from __future__ import annotations

import pandas as pd

from scipy.spatial.distance import pdist, squareform

from dataclasses import dataclass
from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parents[1]
# sys.path.append(str(BASE_DIR))

from params import file_test, upper_bound_default


@dataclass
class DataUtil:
    def __init__(self, file_path):

        self.file_path = file_path
        self.name = 'NAME'
        self.type = 'TYPE'
        self.comment = 'COMMENT'
        self.dimension = 'DIMENSION'
        self.clusters = 'CLUSTERS'
        self.subclusters = 'SUBCLUSTERS'
        self.edge_weight_type = 'EDGE_WEIGHT_TYPE'
        self.node_coord_section = 'NODE_COORD_SECTION'
        self.cgtsp_node_section = 'CGTSP_NODE_SECTION'

        self.data = {}

        self.corr_df = pd.DataFrame(columns=['x', 'y', 'cluster_id', 'subcluster_id'])
        self.distance_df = pd.DataFrame(data=None)

        with open(self.file_path, 'r') as f:
            lines = f.readlines()
        for line in lines:
            if line.startswith(self.name):
                self.data['name'] = line.split(':')[1].strip()
            elif line.startswith(self.type):
                self.data['type'] = line.split(':')[1].strip()
            elif line.startswith(self.comment):
                if 'comments' not in self.data:
                    self.data['comments'] = []
                self.data['comments'].append(line.split(':')[1].strip())
            elif line.startswith(self.dimension):
                self.data['dimension'] = int(line.split(':')[1].strip())
            elif line.startswith(self.clusters):
                self.data['clusters'] = int(line.split(':')[1].strip())
            elif line.startswith(self.subclusters):
                self.data['subclusters'] = int(line.split(':')[1].strip())
            elif line.startswith(self.edge_weight_type):
                self.data['edge_weight_type'] = line.split(':')[1].strip()
            elif line.startswith(self.node_coord_section):
                node_coords = {}
                for i in range(self.data['dimension']):
                    node_id, x, y = map(int, lines[i + 9].split())
                    node_coords[node_id] = (x, y)
                self.data['node_coords'] = node_coords
            elif line.startswith(self.cgtsp_node_section):
                cgtsp_node_section = {}
                for i in range(self.data['dimension']):
                    node_id, cluster_id, subcluster_id = map(int, lines[i + 10 + self.data['dimension']].split())
                    cgtsp_node_section[node_id] = (cluster_id, subcluster_id)
                self.data['cgtsp_node_section'] = cgtsp_node_section

        self.corr_df[['x', 'y']] = pd.DataFrame(list(self.data['node_coords'].values()))
        self.corr_df[['cluster_id', 'subcluster_id']] = pd.DataFrame(list(self.data['cgtsp_node_section'].values()))
        self.corr_df.reset_index(inplace=True)

        # cal distance from node df
        distances = pdist(self.corr_df[['x', 'y']])
        distance_df = squareform(distances)

        self.distance_df = pd.DataFrame(distance_df)


data = DataUtil(file_path=str(BASE_DIR) + file_test)


class Graph:
    def __init__(self, distance_df: pd.DataFrame | None = data.distance_df,
                 corr_df: pd.DataFrame | None = data.corr_df):
        self.distance_df = distance_df
        self.corr_df = corr_df

        self.nodes = list(range(1, len(self.distance_df) + 1))
        self.edges = [(self.distance_df.columns[i], self.distance_df.columns[j]) for i in
                      range(len(self.distance_df.columns))
                      for j in range(i + 1, len(self.distance_df.columns)) if (self.distance_df.iloc[i, j] > 0)]

        self.num_nodes = len(self.nodes)
        self.num_edges = len(self.edges)

        self.cluster_df = self.corr_df.reset_index()[['cluster_id', 'subcluster_id', 'index']].groupby(
            ['cluster_id', 'subcluster_id']).count()

        self.num_clusters = self.corr_df['cluster_id'].nunique()
        self.num_subclusters = self.corr_df.groupby('cluster_id')['subcluster_id'].nunique().sum()


class CGTSP:
    def __init__(self, upper_bound: float | None = upper_bound_default,
                 graph: Graph | None = Graph()):
        if upper_bound is None:
            print('If you are not input upper_bound, this model will using upper bound default')
            self.upper_bound = upper_bound_default
        else:
            self.upper_bound = upper_bound
        self.graph = graph


class TSP_constraint:
    def __init__(self, p1: float, p2: float, graph: Graph | None = Graph()):
        self.p1 = p1
        self.p2 = p2
        self.graph = graph


class TSP:
    def __init__(self, graph: Graph | None = Graph()):
        self.graph = graph


class Tour:
    def __init__(self, cities):
        self.cities = cities

    def get_distance(self):
        """
        Calculate the distance between cities in the tour.
        """
        pass

    def is_eligible(self):
        """
        Check if the tour is eligible for TSP.
        """
        pass


