import pandas as pd
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp


def create_data_model(distance_df):
    data = {}
    # Chuyển đổi DataFrame thành mảng numpy
    distance_matrix = distance_df.values.tolist()
    data['distance_matrix'] = distance_matrix
    data['num_vehicles'] = 1  # Số lượng phương tiện
    data['depot'] = 0  # Điểm xuất phát
    return data


def main(distance_df):
    data = create_data_model(distance_df)
    # Tạo solver
    manager = pywrapcp.RoutingIndexManager(len(data['distance_matrix']),
                                           data['num_vehicles'], data['depot'])
    routing = pywrapcp.RoutingModel(manager)

    # Tạo các bộ chỉ mục để truy cập ma trận khoảng cách
    def distance_callback(from_index, to_index):
        # Lấy khoảng cách giữa hai thành phố
        return data['distance_matrix'][manager.IndexToNode(from_index)][manager.IndexToNode(to_index)]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # Thiết lập thời gian giới hạn tối đa cho mỗi giải pháp
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.time_limit.seconds = 10

    # Tìm lời giải
    solution = routing.SolveWithParameters(search_parameters)

    # In kết quả
    if solution:
        index = routing.Start(0)
        plan_output = 'Lộ trình: '
        route_distance = 0
        while not routing.IsEnd(index):
            plan_output += ' {} ->'.format(manager.IndexToNode(index))
            previous_index = index
            index = solution.Value(routing.NextVar(index))
            route_distance += routing.GetArcCostForVehicle(previous_index, index, 0)
        plan_output += ' {}\n'.format(manager.IndexToNode(index))
        route_distance += routing.GetArcCostForVehicle(previous_index, index, 0)
        plan_output += 'Khoảng cách lộ trình: {} đơn vị'.format(route_distance)
        print(plan_output)

from model_1 import tsp
if __name__ == '__main__':
    # Đọc ma trận khoảng cách từ DataFrame
    distance_df = tsp.graph.distance_df

    main(distance_df)