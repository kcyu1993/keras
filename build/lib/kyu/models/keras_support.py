from keras.layers import SecondaryStatistic, O2Transform, WeightedProbability, Flatten, Dense


def covariance_block_original(input_tensor, nb_class, stage, block, epsilon=0, parametric=[], activation='relu'):
    if epsilon > 0:
        cov_name_base = 'cov' + str(stage) + block + '_branch_epsilon' + str(epsilon)
    else:
        cov_name_base = 'cov' + str(stage) + block + '_branch'
    o2t_name_base = 'o2t' + str(stage) + block + '_branch'
    wp_name_base = 'wp' + str(stage) + block + '_branch'

    x = SecondaryStatistic(name=cov_name_base, eps=epsilon)(input_tensor)
    for id, param in enumerate(parametric):
        x = O2Transform(param, activation='relu', name=o2t_name_base + str(id))(x)
    x = WeightedProbability(nb_class, activation=activation, name=wp_name_base)(x)
    return x


def covariance_block_vector_space(input_tensor, nb_class, stage, block, epsilon=0, parametric=[], activation='relu'):
    if epsilon > 0:
        cov_name_base = 'cov' + str(stage) + block + '_branch_epsilon' + str(epsilon)
    else:
        cov_name_base = 'cov' + str(stage) + block + '_branch'
    dense_name_base = 'dense' + str(stage) + block + '_branch'

    x = SecondaryStatistic(name=cov_name_base, eps=epsilon)(input_tensor)
    x = Flatten()(x)
    for id, param in enumerate(parametric):
        x = Dense(param, activation=activation, name=dense_name_base + str(id))(x)
    x = Dense(nb_class, activation=activation, name=dense_name_base)(x)
    return x