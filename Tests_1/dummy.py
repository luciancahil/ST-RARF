import numpy as np

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

# Not speed. It's an infinute loop.
# Len(test_neighbour) is still 73, but the fully chosen teset thing gets frozen at 44. Why?
# Okay, at the end, 34 is still there in the list , but 34 never gets added. Why?