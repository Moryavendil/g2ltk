from typing import Optional, Any, Tuple, Dict, List
import numpy as np
import cv2 # to manipulate images and videos
import os # to navigate in the directories
import shutil # to remove directories

from tools import display, VERSION
from tools import rivuletfinding, datareading


save_directory:str = 'analysis_files'
index_name:str = 'index'
index_path = os.path.join(save_directory, index_name + '.npz')

def ensure_save_directory_exists():
    if not os.path.isdir(save_directory):
        os.mkdir(save_directory)

def find_filename_without_duplicates(folder:str, name:str, extension:str = 'txt') -> str:
    if extension[0] == '.':
        extension = extension[1:]
    ID = 0
    file_name =  name
    while os.path.exists(os.path.join(folder, file_name + '.' + extension)):
        ID += 1
        file_name =  name + ' - ' + str(ID)
    return file_name + '.' + extension

def generate_appropriate_filename(parameters:Dict[str, Any]) -> str:
    toolsversion_str = 'tools_v'+VERSION
    datatype = parameters.get('datatype', 'unknown-dtype')
    dataset = parameters.get('dataset', None)
    acquisition = parameters.get('acquisition', None)
    name = toolsversion_str + '-' + datatype + (('-' + dataset) if dataset is not None else '') + (('-' + acquisition) if acquisition is not None else '')
    return find_filename_without_duplicates(save_directory, name, extension='npz')


### OVERWRITE THE INDEX OF SAID CATEGORY
def set_index(index:Dict[str, Any]):
    ensure_save_directory_exists()
    np.savez(index_path, index=index)

### CLEAN THE INDEX BY REMOVING UNEXISTENT ENTRIES
def clean_index(index:Dict[str, Any], verbose: int = 1):
    # Find unexistent entries
    bad_items = []
    for item in index.keys(): # run across the items that are indeed in the index
        file_path = os.path.join(save_directory, item)
        if not(os.path.isfile(file_path)):
            bad_items.append(item)

    # Remove the items
    for item in [i for i in bad_items if (i in index)]:  # run across the items that are indeed in the index
        if verbose >= 3:
            print(f'    INFO: Removing item {item} from index since it no longer exists (did you delete it manually?)')
        index.pop(item)
        if verbose >= 4: print(f'\r    DBUG: Removed item {item} from index')
        file_path = os.path.join(save_directory, item)
        if os.path.isfile(file_path): os.remove(file_path)
        if verbose >= 4: print(f'\r    DBUG: Deleted file {file_path}')
    set_index(index)

### OBTAIN THE INDEX OF SAID CATEGORY
def get_index(verbose: int = 1) -> Dict[str, Any]:
    ensure_save_directory_exists()
    if not os.path.isfile(index_path):
        np.savez(index_path, index = {})
    index = np.load(index_path, allow_pickle=True)['index'][()]

    clean_index(index, verbose=verbose)

    return index

### GET THE ITEMS WITH THE CORRESPONDING PARAMETERS
def get_items(parameters: dict, total_match:bool = False, verbose: int = 1) -> List[str]:
    # Open the index
    index = get_index(verbose=verbose)
    items = []
    # search the index
    for candidate_file_name in index.keys():
        candidate_parameters = index[candidate_file_name]
        parameters_match = True
        if total_match:
            parameters_match = candidate_parameters.keys() == parameters.keys()
        else:
            for key in parameters.keys():
                if (key != 'framenumbers') and (key not in candidate_parameters.keys()):
                    parameters_match = False
        if parameters_match:
            same_parameters_values = [np.prod(candidate_parameters[key] == parameters[key]) for key in parameters.keys() if key != 'framenumbers']
            while np.array(same_parameters_values).shape != ():
                same_parameters_values = np.prod(np.array(same_parameters_values))
            if same_parameters_values: # here the file is the one we search
                items.append(candidate_file_name)
    if verbose >= 4:
        print(f'\r  Found {len(items)} item(s) with {"totally" if total_match else "partially"} matching parameter.')
    if verbose >= 5:
        print(f'\r    Items: {items}')
    return items

def framenumbers_available(parameters: dict, verbose: int = 1) -> Optional[np.ndarray]:
    if verbose >= 4:
        print(f'\rFetching saved framenumbers for {parameters["acquisition"]} ({parameters["dataset"]})')
    # get the items
    items = get_items(parameters, total_match=False, verbose=verbose)

    # get the index
    index = get_index(verbose=verbose)

    # Find the good item
    for item in items: # run across the items that are indeed in the index
        file_path = os.path.join(save_directory, item)
        if os.path.isfile(file_path):
            if verbose >= 4: print(f'\rFetching a number of framenumbers from "{item}"')
            framenumbers = index[item].get('framenumbers', np.empty(0, int))
            return framenumbers # return the first one
        else:
            if verbose >=3: print(f'\rFile {file_path} does not exist ?!')


    return np.empty(0, int)

### GETS THE DATA OF SAID CATEGORY CORRESPONDING TO SAID CONDITION
def fetch_saved_data(parameters: dict, verbose: int = 1) -> Any:
    if verbose >= 4:
        if parameters.get("dataset", None) is not None:
            if parameters.get("acquisition", None) is not None:
                print(f'\rFetching saved data for {parameters["acquisition"]} ({parameters["dataset"]})')
            else:
                print(f'\rFetching saved data ({parameters["dataset"]})')
        else:
            print(f'\rFetching saved data')

    # get the items
    items = get_items(parameters, total_match=False, verbose=verbose)
    # Find the good item
    for item in items: # run across the items that are indeed in the index
        file_path = os.path.join(save_directory, item)
        if os.path.isfile(file_path):
            if verbose >= 3:
                print(f'\rFetching data from "{item}"')
            return np.load(file_path, allow_pickle=True)['data'][()] # return the first one
    return None

### ERASE ALL RECORDINGS OF SAID CATEGORY CORRESPONDING TO SAID CONDITION
def erase_items(parameters: dict, total_match:bool = False, verbose: int = 1) -> None:
    """

    :param parameters:
    :param total_match:
    :param verbose:
    :return:
    """
    if parameters is None: return None
    if verbose >= 4: print(f'\r  Removing items')
    # open the index
    index = get_index(verbose=verbose)
    # get the items to remove
    items = get_items(parameters=parameters, total_match=total_match, verbose=verbose)
    # Remove the items
    for item in [i for i in items if (i in index)]:  # run across the items that are indeed in the index
        index.pop(item)
        if verbose >= 4: print(f'\r    Removed item {item} from index')
        file_path = os.path.join(save_directory, item)
        if os.path.isfile(file_path): os.remove(file_path)
        if verbose >= 4: print(f'\r    Deleted file {file_path}')
    # save the modified index
    set_index(index)

### ERASE ALL RECORDINGS OF SAID CATEGORY CORRESPONDING TO SAID CONDITION
def erase_all(verbose: int = 1) -> None:
    raise(Exception('Function unimplemented'))

### ADD A FILE (ITEM) TO THE INDEX, IDENTIFIED BY ITS PARAMETERS
def add_item_to_index(item:str, parameters: dict, remove_similar_older_entries:bool = False, verbose:int = 1) -> None:
    if parameters is None:return None
    if remove_similar_older_entries:
        erase_items(parameters=parameters, total_match=False, verbose=verbose)
    # Open the index
    index = get_index(verbose=verbose)
    # Register the file (item)
    index[item] = parameters
    # Save the updated index
    set_index(index)
    if verbose >= 2: print(f'\r  Added item {item} to index')

### SAVES THE DATA OF SAID CATEGORY CORRESPONDING TO SAID CONDITION
def save_data(data: Any, parameters: dict, verbose:int = 1) -> None:
    if parameters is None:return None
    if verbose >= 2:
        print(f'\rSaving an item...', end='')
    # generate the filename / item name
    filename = generate_appropriate_filename(parameters)
    # save the new index
    add_item_to_index(filename, parameters, remove_similar_older_entries=True, verbose=verbose)
    # save the file
    np.savez(os.path.join(save_directory, filename), data=data)
    # end
    if verbose >= 2:
        print(f'\rSaved an item named {filename}')

### TO GENERATE DATA
def data_generating_fn(parameters:Dict[str, Any], verbose:int=1):
    datatype = parameters.get('datatype', None)
    if datatype is None:
        # todo warning here
        print('WARNING: Datatype is None ?!')
        return None
    elif datatype == 'cos':
        if (verbose >= 4): print('generating datatype: COS')
        return rivuletfinding.find_cos(**parameters)
    elif datatype == 'borders':
        if (verbose >= 4): print('generating datatype: BORDERS')
        return rivuletfinding.find_borders(**parameters)
    elif datatype == 'bol':
        if (verbose >= 4): print('generating datatype: BOL')
        return rivuletfinding.find_bol(**parameters)
    elif datatype == 'acqu_freq_from_chronos_mp4_timestamps': #FIXME DEPRECIATED
        if (verbose >= 4): print('generating datatype: acqu_freq_from_chronos_mp4_timestamps')
        return datareading.get_chronos_acquisition_frequency_from_mp4_video(**parameters)
    else:
        # todo error here
        print(f'ERROR: datatype not understood: {datatype}')
        return None

def fetch_or_generate_data(datatype: str, dataset:str, acquisition:str, verbose:int = 1, **kwargs):
    parameters = {'datatype': datatype, 'dataset': dataset, 'acquisition': acquisition, **kwargs}
    parameters['framenumbers'] = parameters.get('framenumbers', None)
    available_fns = framenumbers_available(parameters, verbose=verbose)
    wanted_fns = parameters['framenumbers']
    if verbose >= 4:
        print(f'\rWanted framenumbers "{wanted_fns}"')
        print(f'\rAvailable framenumbers "{available_fns}"')
    if available_fns is None: # si tout est dispo
        if verbose >= 4:print(f'\rAll framenumbers available')
        data = fetch_saved_data(parameters, verbose=verbose) # prendre tout
        if wanted_fns is None: # si on demande tout
            return data # donner tout
        else: # si on demande une partie
            return data[wanted_fns] # donner une partie
    else: # si une partie seulement est dispo
        if wanted_fns is None: # si on demande tout
            data = data_generating_fn(parameters, verbose=verbose) # on calcule
            save_data(data, parameters, verbose=verbose) # on sauvegarde
            return fetch_saved_data(parameters, verbose=verbose) # et on va chercher
        else: # si on demande une partie
            if np.sum(np.isin(available_fns, wanted_fns, assume_unique=True)) == len(wanted_fns): # si cette partie est incluse dans la partie disponible
                if verbose >= 4:print(f'\r{len(available_fns)} framenumbers available, {len(wanted_fns)} wanted. No need to generate data.')
                data = fetch_saved_data(parameters, verbose=verbose) # prendre tout
                return data[np.isin(available_fns, wanted_fns, assume_unique=True)] # donner une partie
            else: # si la partie demandée on l'a pas
                if verbose >= 4:print(f'\r{len(available_fns)} framenumbers available, {len(wanted_fns)} wanted. Will have to generate data.')
                total_fns = np.sort( np.unique( np.concatenate((wanted_fns, available_fns)) ) ) # on prend l'union des deux
                if verbose >= 4:print(f'\rFramenumbers generated: "{total_fns}"')
                parameters['framenumbers'] = total_fns
                data = data_generating_fn(parameters, verbose=verbose) # on calcule
                save_data(data, parameters, verbose=verbose) # on sauvegarde
                return fetch_saved_data(parameters, verbose=verbose) # et on va chercher

def fetch_or_generate_data_from_parameters(datatype: str, parameters):
    parameters['datatype'] = datatype
    parameters['framenumbers'] = parameters.get('framenumbers', None)
    verbose = parameters.get('verbose', 1)
    available_fns = framenumbers_available(parameters, verbose=verbose)
    wanted_fns = parameters['framenumbers']
    if verbose >= 4:
        print(f'\rWanted framenumbers "{wanted_fns}"')
        print(f'\rAvailable framenumbers "{available_fns}"')
    if available_fns is None: # si tout est dispo
        if verbose >= 4:print(f'\rAll framenumbers available')
        data = fetch_saved_data(parameters, verbose=verbose) # prendre tout
        if wanted_fns is None: # si on demande tout
            return data # donner tout
        else: # si on demande une partie
            return data[wanted_fns] # donner une partie
    else: # si une partie seulement est dispo
        if wanted_fns is None: # si on demande tout
            data = data_generating_fn(parameters, verbose=verbose) # on calcule
            save_data(data, parameters, verbose=verbose) # on sauvegarde
            return fetch_saved_data(parameters, verbose=verbose) # et on va chercher
        else: # si on demande une partie
            if np.sum(np.isin(available_fns, wanted_fns, assume_unique=True)) == len(wanted_fns): # si cette partie est incluse dans la partie disponible
                if verbose >= 4:print(f'\r{len(available_fns)} framenumbers available, {len(wanted_fns)} wanted. No need to generate data.')
                data = fetch_saved_data(parameters, verbose=verbose) # prendre tout
                return data[np.isin(available_fns, wanted_fns, assume_unique=True)] # donner une partie
            else: # si la partie demandée on l'a pas
                if verbose >= 4:print(f'\r{len(available_fns)} framenumbers available, {len(wanted_fns)} wanted. Will have to generate data.')
                total_fns = np.sort( np.unique( np.concatenate((wanted_fns, available_fns)) ) ) # on prend l'union des deux
                if verbose >= 4:print(f'\rFramenumbers generated: "{total_fns}"')
                parameters['framenumbers'] = total_fns
                data = data_generating_fn(parameters, verbose=verbose) # on calcule
                save_data(data, parameters, verbose=verbose) # on sauvegarde
                return fetch_saved_data(parameters, verbose=verbose) # et on va chercher
