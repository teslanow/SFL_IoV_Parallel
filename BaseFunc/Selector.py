import numpy as np
def uniform_select(origin_list, num_to_select):
    selected_list = np.random.choice(origin_list, size=num_to_select, replace=False).tolist()
    return selected_list