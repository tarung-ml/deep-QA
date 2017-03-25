from util import one_hot
def load_data(data_dir, data_subset):
    path = data_dir + "/" + data_subset
    Q, P, A_start, A_end, P_raw, A_raw, Q_len, P_len, A_len= [], [], [], [], [], [], [], [], []
    with open(path + ".ids.question") as f:
        for line in f:
            Q.append(line.split())
            Q_len.append(len(line.split()))
    with open(path + ".ids.context") as f:
        for line in f:
            P.append(line.split())
            P_len.append(len(line.split()))
    with open(path + ".span") as f:
        for line in f:
            start_index, end_index =  [int(index) for index in  line.split()]
            A_start.append(one_hot(start_index + 1, start_index)) # [0, 0, ..., start_index] will be padded  to 1-hot vec of max_answer_size i.e. max_context_size
            A_end.append(one_hot(end_index + 1, end_index))  # [0, 0, ..., end_index] will be padded  to a 1-hot vec of max_answer_size i.e. max_context_size
            A_len.append(end_index - start_index + 1)
    with open(path + ".context") as f:
        for line in f:
            P_raw.append(line)
    with open(path + ".answer") as f:
        for line in f:
            A_raw.append(line)
    return Q, P, A_start, A_end, A_len, P_raw, A_raw, Q_len, P_len


def load_data_home(path):
    Q, P, A_start, A_end, P_raw, A_raw, Q_len, P_len, A_len= [], [], [], [], [], [], [], [], []
    with open(path + ".ids.question") as f:
        for line in f:
            Q.append(line.split())
            Q_len.append(len(line.split()))
    with open(path + ".ids.context") as f:
        for line in f:
            P.append(line.split())
            P_len.append(len(line.split()))
    with open(path + ".span") as f:
        for line in f:
            start_index, end_index =  [int(index) for index in  line.split()]
            A_start.append(one_hot(start_index + 1, start_index)) # [0, 0, ..., start_index] will be padded  to 1-hot vec of max_answer_size i.e. max_context_size
            A_end.append(one_hot(end_index + 1, end_index))  # [0, 0, ..., end_index] will be padded  to a 1-hot vec of max_answer_size i.e. max_context_size
            A_len.append(end_index - start_index + 1)
    with open(path + ".context") as f:
        for line in f:
            P_raw.append(line)
    with open(path + ".answer") as f:
        for line in f:
            A_raw.append(line)
    return Q, P, A_start, A_end, A_len, P_raw, A_raw, Q_len, P_len
