def get_train_test_data(name, token, p):
    session_path = r'datasets/datasets_origin/{}/session.csv'.format(name)
    neigh_path = r'datasets/datasets_origin/{}/neighbor.csv'.format(name)
    session_datas = open(session_path, 'r').readlines()
    neigh_datas = open(neigh_path, 'r').readlines()

    # get all_item_dict to instead item's id
    all_item_dict = {}
    item_order = 0
    for data in session_datas:
        if name == 'Amazon_Books':
           data = data.split('\n')[0].split('|')[0].split(',')  # Amazon_Books
        else:
           data = data.split('\n')[0].split(',')              # merge and yoochoose

        item_list = data[1:]
        for item_id in item_list:
            if item_id not in all_item_dict:
               all_item_dict[item_id] = item_order
               item_order += 1

    # user_item_num = len(all_item_dict)

    for data in neigh_datas:
        data = data.split('\n')[0].split(',')
        item_list = data[2:]
        for item_id in item_list:
            if item_id not in all_item_dict:
                all_item_dict[item_id] = item_order
                item_order += 1

    # get user_session_dict
    user_session_dict = {}
    max_len_session = 0
    for data in session_datas:
        if name == 'Amazon_Books':
            data = data.split('\n')[0].split('|')[0].split(',')  # Amazon_Books
        else:
            data = data.split('\n')[0].split(',')  # merge and yoochoose

        user_id = data[0]
        user_session_dict[user_id] = [all_item_dict[i] for i in data[1:]]
        if len(data[1:]) > max_len_session:
           max_len_session = len(data[1:])

    # get user_neigh_dict
    user_neigh_dict = {}
    max_len_neigh = 0
    for data in neigh_datas:
        data = data.split('\n')[0].split(',')
        user_id = data[0]
        neigh_id = data[1]
        if user_id not in user_neigh_dict:
            user_neigh_dict[user_id] = {neigh_id: [all_item_dict[i] for i in data[2:]]}
        else:
            user_neigh_dict[user_id][neigh_id] = [all_item_dict[i] for i in data[2:]]
        if len(data[2:]) > max_len_neigh:
            max_len_neigh = len(data[2:])

    item_num = len(all_item_dict)
    session_m = []
    neigh_m = []
    for user_id in user_session_dict:
        session_list = user_session_dict[user_id]
        session_m.append(session_list)
        per_user_neigh_list = []
        for neigh_id in user_neigh_dict[user_id]:
            neigh_list = user_neigh_dict[user_id][neigh_id]
            per_user_neigh_list.append(neigh_list)
        neigh_m.append(per_user_neigh_list)

    split_pos = int(len(user_session_dict) * p)
    train_data = session_m[:split_pos]; test_data = session_m[split_pos:]
    train_neigh = neigh_m[:split_pos]; test_neigh = neigh_m[split_pos:]

    for pos, neigh_list in enumerate(train_neigh):
        while len(neigh_list) < 5:
            neigh_list += neigh_list
        neigh_list = neigh_list[:5]
        train_neigh[pos] = neigh_list

    for pos, neigh_list in enumerate(test_neigh):
        while len(neigh_list) < 5:
            neigh_list += neigh_list
        neigh_list = neigh_list[:5]
        test_neigh[pos] = neigh_list

    return train_data, train_neigh, test_data, test_neigh, item_num, max_len_neigh, max_len_session
    # return train_data, train_neigh, test_data, test_neigh, user_item_num, max_len_neigh, max_len_session


