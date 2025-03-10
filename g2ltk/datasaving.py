from typing import Optional, Any, Dict, List
import numpy as np
import os # to navigate in the directories
# import shutil # to remove directories

from g2ltk import log_trace, log_debug, log_subinfo, log_info, log_warning, log_error, VERSION
from g2ltk import rivuletfinding, datareading


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
    dataset = str(parameters.get('dataset', None))
    acquisition = str(parameters.get('acquisition', None))
    name = toolsversion_str + '-' + datatype + (('-' + dataset) if dataset is not None else '') + (('-' + acquisition) if acquisition is not None else '')
    return find_filename_without_duplicates(save_directory, name, extension='npz')


### OVERWRITE THE INDEX OF SAID CATEGORY
def set_index(index:Dict[str, Any]):
    ensure_save_directory_exists()
    np.savez(index_path, index=index)

### CLEAN THE INDEX BY REMOVING UNEXISTENT ENTRIES
def clean_index(index:Dict[str, Any], verbose:Optional[int]=None):
    # Find unexistent entries
    bad_items = []
    for item in index.keys(): # run across the items that are indeed in the index
        file_path = os.path.join(save_directory, item)
        if not(os.path.isfile(file_path)):
            bad_items.append(item)

    # Remove the items
    for item in [i for i in bad_items if (i in index)]:  # run across the items that are indeed in the index
        log_info(f'Removing item {item} from index since it no longer exists (did you delete it manually?)', verbose)
        index.pop(item)
        log_debug(f'Removed item {item} from index', verbose)
        file_path = os.path.join(save_directory, item)
        if os.path.isfile(file_path): os.remove(file_path)
        log_debug(f'Deleted file {file_path}', verbose)
    set_index(index)

### OBTAIN THE INDEX OF SAID CATEGORY
def get_index(verbose:Optional[int]=None) -> Dict[str, Any]:
    ensure_save_directory_exists()
    if not os.path.isfile(index_path):
        np.savez(index_path, index = {})
    index = np.load(index_path, allow_pickle=True)['index'][()]

    clean_index(index, verbose=verbose)

    return index

### GET THE ITEMS WITH THE CORRESPONDING PARAMETERS
def get_items(parameters: dict, total_match:bool = False, verbose:Optional[int]=None) -> List[str]:
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
                if (key not in ['framenumbers', 'verbose']) and (key not in candidate_parameters.keys()):
                    parameters_match = False
        if parameters_match:
            same_parameters_values = [np.prod(candidate_parameters[key] == parameters[key]) for key in parameters.keys() if (key not in ['framenumbers', 'verbose'])]
            while np.array(same_parameters_values).shape != ():
                same_parameters_values = np.prod(np.array(same_parameters_values))
            if same_parameters_values: # here the file is the one we search
                items.append(candidate_file_name)
    log_debug(f'Found {len(items)} item(s) with {"totally" if total_match else "partially"} matching parameter.', verbose)
    log_trace(f'Items: {items}', verbose)
    return items

def framenumbers_available(parameters: dict, verbose:Optional[int]=None) -> Optional[np.ndarray]:
    log_debug(f'Searching for saved framenumbers for {parameters["acquisition"]} ({parameters["dataset"]})', verbose)
    # get the items
    items = get_items(parameters, total_match=False, verbose=verbose)

    # get the index
    index = get_index(verbose=verbose)

    # Find the good item
    for item in items: # run across the items that are indeed in the index
        file_path = os.path.join(save_directory, item)
        if os.path.isfile(file_path):
            log_trace(f'Fetching a number of framenumbers from "{item}"', verbose)
            framenumbers = index[item].get('framenumbers', np.empty(0, int))
            return framenumbers # return the first one
        else:
            log_warning(f'File {file_path} does not exist ?!', verbose)


    return np.empty(0, int)

### GETS THE DATA OF SAID CATEGORY CORRESPONDING TO SAID CONDITION
def fetch_saved_data(parameters: dict, verbose:Optional[int]=None) -> Any:
    log_debug(f'Fetching {parameters.get("datatype", "data")} for {parameters.get("acquisition", "???")} ({parameters.get("dataset", "???")})', verbose)

    # get the items
    items = get_items(parameters, total_match=False, verbose=verbose)
    # Find the good item
    for item in items: # run across the items that are indeed in the index
        file_path = os.path.join(save_directory, item)
        if os.path.isfile(file_path):
            log_trace('Fetching data from file "{item}"', verbose)
            return np.load(file_path, allow_pickle=True)['data'][()] # return the first one

    log_warning(f'Did not find the right stuff ?')
    return None

### ERASE ALL RECORDINGS OF SAID CATEGORY CORRESPONDING TO SAID CONDITION
def erase_items(parameters: dict, total_match:bool = False, verbose:Optional[int]=None) -> None:
    """

    :param parameters:
    :param total_match:
    :param verbose:
    :return:
    """
    if parameters is None: return None
    log_trace(f'Removing items', verbose)
    # open the index
    index = get_index(verbose=verbose)
    # get the items to remove
    items = get_items(parameters=parameters, total_match=total_match, verbose=verbose)
    # Remove the items
    for item in [i for i in items if (i in index)]:  # run across the items that are indeed in the index
        index.pop(item)
        log_trace(f'    Removed item {item} from index', verbose)
        file_path = os.path.join(save_directory, item)
        if os.path.isfile(file_path): os.remove(file_path)
        log_trace(f'    Deleted file {file_path}', verbose)
    # save the modified index
    set_index(index)

### ERASE ALL
def erase_all(verbose:Optional[int]=None) -> None:
    log_error('Function unimplemented')

### ERASE ALL RECORDINGS OF SAID CATEGORY CORRESPONDING TO SAID CONDITION
def nuke_all(verbose:Optional[int]=None) -> None:
    log_info(f'Nuking EVERYTHING', verbose)
    for file in os.listdir(save_directory):
        if os.path.isfile(os.path.join(save_directory, file)): os.remove(os.path.join(save_directory, file))
    os.rmdir(save_directory)

### ADD A FILE (ITEM) TO THE INDEX, IDENTIFIED BY ITS PARAMETERS
def add_item_to_index(item:str, parameters: dict, remove_similar_older_entries:bool = False, verbose:Optional[int]=None) -> None:
    if parameters is None:return None
    if remove_similar_older_entries:
        erase_items(parameters=parameters, total_match=False, verbose=verbose)
    # Open the index
    index = get_index(verbose=verbose)
    # Register the file (item)
    index[item] = parameters
    # Save the updated index
    set_index(index)
    log_trace(f'Added item {item} to index', verbose)

### SAVES THE DATA OF SAID CATEGORY CORRESPONDING TO SAID CONDITION
def save_data(data: Any, parameters: dict, verbose:Optional[int]=None) -> None:
    if parameters is None:
        log_warning('(save_data) parameters are None')
        return None
    if data is None:
        log_warning('(save_data) data is None')
        return None
    log_trace(f'Saving an item', verbose)
    # generate the filename / item name
    filename = generate_appropriate_filename(parameters)
    # save the new index
    add_item_to_index(filename, parameters, remove_similar_older_entries=True, verbose=verbose)
    # save the file
    np.savez(os.path.join(save_directory, filename), data=data)
    # end
    log_trace(f'Saved an item named {filename}', verbose)

### TO GENERATE DATA
def data_generating_fn(parameters:Dict[str, Any], verbose:int=1):
    datatype = parameters.get('datatype', None)
    log_debug(f'Generating data: {datatype}', verbose)
    if datatype is None:
        log_error('datatype is None ?!', verbose)
        return None
    elif datatype == 'cos':
        return rivuletfinding.find_cos(**parameters)
    elif datatype == 'borders':
        return rivuletfinding.find_borders(**parameters)
    elif datatype == 'fwhmol':
        return rivuletfinding.find_fwhmol(**parameters)
    elif datatype == 'bol':
        return rivuletfinding.find_bol(verbose=verbose, **parameters)
    else:
        log_error(f'datatype not understood: {datatype}', verbose)
        return None

def fetch_or_generate_data(datatype:str, dataset:str, acquisition:str, verbose:Optional[int]=None, **kwargs):
    parameters = {'dataset': dataset, 'acquisition': acquisition, **kwargs}

    return fetch_or_generate_data_from_parameters(datatype, parameters, verbose=verbose)

def fetch_or_generate_data_from_parameters(datatype:str, parameters:dict, verbose:int = None):
    log_subinfo(f'Fetching or generating data: {datatype}', verbose)
    log_trace(f'Parameters: {parameters}', verbose)

    parameters['datatype'] = datatype

    # petite rustine ponctuelle pour gagner du temps
    if datatype == 'bol':
        fetch_or_generate_data_from_parameters('borders', {**parameters}, verbose=verbose)

    parameters['framenumbers'] = parameters.get('framenumbers', None)
    available_fns = framenumbers_available(parameters, verbose=verbose)
    wanted_fns = parameters['framenumbers']
    log_trace(f'Wanted framenumbers "{wanted_fns}"', verbose)
    log_trace(f'Available framenumbers "{available_fns}"', verbose)
    if available_fns is None: # si tout est dispo
        log_debug(f'All framenumbers available', verbose)
        data = fetch_saved_data(parameters, verbose=verbose) # prendre tout
        if wanted_fns is None: # si on demande tout
            return data # donner tout
        else: # si on demande une partie
            return data[wanted_fns] # donner une partie

    else: # Si tout n'est pas dispo
        if wanted_fns is not None: # si on demande pas tout
            if np.sum(np.isin(available_fns, wanted_fns, assume_unique=True)) == len(wanted_fns): # si cette partie est incluse dans la partie disponible
                log_debug(f'{len(available_fns)} framenumbers available, {len(wanted_fns)} wanted. No need to generate data.', verbose)
                data = fetch_saved_data(parameters, verbose=verbose) # prendre tout
                return data[np.isin(available_fns, wanted_fns, assume_unique=True)] # donner une partie
            else: # si on a des trucs mais la partie demandée on l'a pas
                log_debug(f'{len(available_fns)} framenumbers available, {len(wanted_fns)} wanted.', verbose)
                total_fns = np.sort( np.unique( np.concatenate((wanted_fns, available_fns)) ) ) # on prend l'union des deux
        else: # si on demande tout
            log_debug('All framenumbers wanted', verbose)
            total_fns = None

        # Ici on va devoir générer total_fns.
        log_debug('We need to generate data', verbose)
        total_fns_explicit = total_fns
        if total_fns is None:
            acquisition_path = rivuletfinding.get_acquisition_path_from_parameters(**parameters)
            total_fns_explicit = np.arange(datareading.get_number_of_available_frames(acquisition_path))
        log_debug(f'Data to generate: {len(total_fns_explicit)} frames', verbose)

        chunk_size:int = 500

        # on y va chunk_size par chunk_size
        number_of_chunks:int = len(total_fns_explicit)//chunk_size+(len(total_fns_explicit)%chunk_size != 0)
        log_debug(f'Splitting data into {number_of_chunks} chunks of {chunk_size} frames', verbose)
        # les chunk_size premiers
        parameters['framenumbers'] = total_fns_explicit[:chunk_size]
        log_info(f'Generating {datatype} (chunk 1/{number_of_chunks})', verbose)
        data = data_generating_fn(parameters, verbose=verbose) # on calcule
        if data is None:log_warning('The data generated is None', verbose)
        # puis on refait le calcul par groupe de 500
        for i in range(1, number_of_chunks):
            log_info(f'Generating {datatype} (chunk {i+1}/{number_of_chunks})', verbose)
            parameters['framenumbers'] = total_fns_explicit[i*chunk_size:(i+1)*chunk_size]
            data = np.concatenate((data, data_generating_fn(parameters, verbose=verbose))) # on calcule

        # Maintenant qu'on a le tout, on range bien
        parameters['framenumbers'] = total_fns
        save_data(data, parameters, verbose=verbose) # on sauvegarde

        # maintenant qu'on a généré tout ce qu'il fallait, on va chercher (petite récursion)
        parameters['framenumbers'] = wanted_fns
        return fetch_or_generate_data_from_parameters(datatype, parameters, verbose=verbose)

