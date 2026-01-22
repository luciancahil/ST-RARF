import sys
sys.path.append('../../src/')
import RaRFRegressor
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error
import seaborn as sns
import utils


COLORA = '#027F80'
COLORB = '#B2E5FC'
SEED = 314159


def run_and_plot(radius, X_train, y_train, X_test, y_test, distances, title_extra = None):
    radius_pred, train_neighbours = RaRFRegressor.RaRFRegressor(radius=radius, metric='jaccard', seed=SEED).train_parallel(X_train,y_train, include_self='True')
    radius_testpred, test_neighbours, test_neighbours_list = RaRFRegressor.RaRFRegressor(radius=radius,metric='jaccard', seed=SEED).predict_parallel(X_train, y_train, X_test, distances)    

    test_neighbours = np.array(test_neighbours)
    nan_indexes = []
    index = -1
    for prediction in radius_testpred:
        index +=1
        if np.isnan(prediction) == True:
            nan_indexes.append(index)
        
    radius_testpred_temp = np.delete(radius_testpred,nan_indexes)
    y_test_temp = np.delete(y_test,nan_indexes)


    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(10,5), dpi=300)


    ax1.plot(y_train,y_train, color='grey', zorder=0)

    train_r2 = r2_score(y_train,radius_pred)
    test_r2 = r2_score(y_test_temp,radius_testpred_temp)
    ax1.scatter(y_train,radius_pred, label='train R2 ' + str(round(train_r2,2)), color='#279383')
    ax1.scatter(y_test_temp,radius_testpred_temp, label='test R2 ' + str(round(test_r2,2)), color='white', edgecolor='#279383')

    ax1.set_xlabel('Measured $\Delta\Delta G^‡$ (kcal/mol)')
    ax1.set_ylabel('Predicted $\Delta\Delta G^‡$ (kcal/mol)')
    ax1.legend()


    ax2 = sns.kdeplot(data=[[train_neighbours[x] for x in np.nonzero(train_neighbours)[0]], [test_neighbours[x] for x in np.nonzero(test_neighbours)[0]]], palette=[COLORA, COLORB])
    ax2.legend(['train', 'test'])
    ax2.set_xlim(-10,200)
    ax2.set_xlabel('# of neighbours')

    fig.suptitle(f'Radius {radius}, {len(nan_indexes)}/{len(radius_testpred)} NaNs {title_extra if title_extra else ""}')
    plt.tight_layout()
    plt.show()

    flattened_list = [item for sublist in test_neighbours_list for item in sublist]

    reaction_set = set(flattened_list)


    return (mean_absolute_error(y_test_temp,radius_testpred_temp)), len(nan_indexes), np.average([test_neighbours[x] for x in np.nonzero(test_neighbours)[0]]), reaction_set, test_neighbours_list, test_r2




def get_sorted_neighbours(cutoff, distances): 
    num_test = distances.shape[0]

    
    mask = distances <= cutoff


    
    # the ith element is a list of all neighbours for the ith test example.
    test_neighbours = [np.where(row)[0].tolist() for row in mask]
    
    # the ith element is a list of all neighbours for the ith train example.
    train_neighbours = [np.where(row)[0].tolist() for row in mask.T]
    
    # a list of training examples we've chosen.
    chosen_train = []
    
    
    # number of neighbours already selected for each test example
    num_selected_neighbours = np.array([0] * num_test)
    
    
    # set of test examples that have no more unselected neighbours
    fully_chosen_tests = set([i for i, neighs in enumerate(test_neighbours) if len(neighs) == 0])

    buffer = False
    while (len(fully_chosen_tests) < num_test):

        live_tests = np.array([i for i in range(num_test) if i not in fully_chosen_tests])
    
        # of the test examples that still have unselected neighbours, what is the fewest number of selected neighbours
        fewest_neighs = np.min(num_selected_neighbours[live_tests])
    
        # test examples that have the fewest selected neighbours that still have neighbours to select
        neediest_tests = np.array([t for t in np.where(num_selected_neighbours == fewest_neighs)[0] if t in live_tests])
        cur_mask = mask[neediest_tests]
        
        train_neighbours = [(i, np.where(row)[0].tolist()) for i, row in enumerate(cur_mask.T)]
        train_neighbours = sorted(train_neighbours, key=lambda example: len(example[1]), reverse=True)

        
        # the train example we are adding
        selected = train_neighbours[0][0]
        
        # test_examples that got a new neighbour
        new_neighbours = np.where(mask[:,selected])[0]
        
        chosen_train.append(selected)
        
        # remove the selected reaction from the list of neighbours that have it.
        for neigh in new_neighbours:
            test_neighbours[neigh].remove(selected)
            num_selected_neighbours[neigh] += 1
        
        fully_chosen_tests = set([i for i, neighs in enumerate(test_neighbours) if len(neighs) == 0])
        
        mask[:,selected] = False

    
    return chosen_train




def run_MT(cutoff, max_budget, ref_mae, ref_r2, X_train, y_train, X_test, y_test):
    results = []
    
    distances = utils.get_distances(X_train,X_test)

    sorted_neigh = get_sorted_neighbours(cutoff, distances)
    
    multipliers = np.divide(range(1, 11), 10)
    
    for multiplier in multipliers:
        budget = int(max_budget * multiplier)
        cur_indicies = sorted_neigh[0:budget]
    
        cur_distances = np.full_like(distances, 2)
    
        cur_distances[:,cur_indicies] = distances[:,cur_indicies]
    
        results.append(run_and_plot(cutoff, X_train, y_train, X_test, y_test, cur_distances,  
                                    title_extra = ", Budget: {}, Percent Used: {:.1f}%".format(budget, multiplier * 100)))


    
    cur_RaRF_mae, cur_nans, cur_avg_neighbours, cur_all_reactions, cur_reaction_list, cur_test_r2s = zip(*results)

    print([len(ar) for ar in cur_all_reactions])
    print([int(max_budget * m) for m in multipliers])
    
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(10,5), dpi=300)

    title = "MAE to Percent of Reactions Used \n(Radius = {})".format(cutoff)

    ref_num_points = 1000

    ref_x = np.array(range(0, ref_num_points)) * 100 / ref_num_points

    x_label = "% of Reactions Used\nvs. Single Target"

    ax1.set_title(title)
    ax1.set_ylabel("MAE")
    ax1.set_xlabel(x_label)
    ax1.scatter(ref_x, [ref_mae] * ref_num_points, label = "Reference", s=0.5)
    ax1.scatter(100 * multipliers, cur_RaRF_mae, label="Actual")
    ax1.legend()

    ax2.set_title("MAE to Percent of Reactions Used \n(Radius = 0.1)")
    ax2.yaxis.set_label_position("right")
    ax2.yaxis.tick_right()

    ax2.set_ylabel("R^2",rotation=270)
    ax2.set_xlabel(x_label)
    ax2.scatter(ref_x, [ref_r2] * ref_num_points, label = "Reference", s=0.5)
    ax2.scatter(100 * multipliers, cur_test_r2s, label="Actual")

    ax2.legend()


    plt.show()