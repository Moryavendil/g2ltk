from typing import Optional, Any, Dict, List
import numpy as np
import os

from g2ltk import __version__
from g2ltk import customlog
# customlog has log_error, log_warning, log_info, log_subinfo, log_debug, log_trace and log_subtrace

# WE DO NOT LOAD THESE< SO USING WITHA  VIDEO WILL RAISE AN ERROR...
# from g2ltk import videoreading
# from g2ltk.rivulets import rivuletfinding

save_directory: str = '.datacache'
index_name: str = 'index'
index_path = os.path.join(save_directory, index_name + '.npz')


def ensure_save_directory_exists():
    """
    Create the save directory if it does not already exist.

    Notes
    -----
    Uses the module-level `save_directory` variable.
    Does nothing if the directory already exists.
    """
    if not os.path.isdir(save_directory):
        os.mkdir(save_directory)


def find_filename_without_duplicates(folder: str, name: str, extension: str = 'txt') -> str:
    """
    Return a filename that does not collide with any existing file in `folder`.

    If `name.extension` already exists, appends ` - 1`, ` - 2`, … until a
    free name is found.

    Parameters
    ----------
    folder : str
        Directory in which to search for existing files.
    name : str
        Base filename (without extension).
    extension : str, optional
        File extension, with or without a leading dot (default ``'txt'``).

    Returns
    -------
    str
        A filename (base + extension, **no directory prefix**) guaranteed not
        to exist in `folder` at the time of the call.

    Notes
    -----
    There is an inherent TOCTOU race between this check and the actual file
    creation — not a concern for single-process use.
    """
    if extension[0] == '.':
        extension = extension[1:]
    ID = 0
    file_name = name
    while os.path.exists(os.path.join(folder, file_name + '.' + extension)):
        ID += 1
        file_name = name + ' - ' + str(ID)
    return file_name + '.' + extension


def generate_appropriate_filename(parameters: Dict[str, Any]) -> str:
    """
    Build a versioned, human-readable filename for a data file.

    The name encodes the toolkit version, datatype, dataset, and acquisition
    so that saved files are self-describing.  A duplicate-avoiding suffix is
    appended when needed (see `find_filename_without_duplicates`).

    Parameters
    ----------
    parameters : dict
        Must contain at least ``'datatype'``.  Optionally ``'dataset'`` and
        ``'acquisition'`` are appended to the stem when present and not
        ``None``.

    Returns
    -------
    str
        Filename (including ``.npz`` extension, **no directory prefix**).

    Notes
    -----
    .. warning::
        ``parameters.get('dataset', None)`` and ``parameters.get('acquisition', None)``
        are fetched *before* being cast to ``str``, so the ``None`` guard
        works correctly. Previous versions cast first, making the guard
        always ``True``.
    """
    toolsversion_str = 'g2ltk_v' + __version__
    datatype = parameters.get('datatype', 'unknown-dtype')

    # SANITY: fetch raw values first so the None check is meaningful
    dataset_raw = parameters.get('dataset', None)
    acquisition_raw = parameters.get('acquisition', None)

    dataset_part = ('-' + str(dataset_raw)) if dataset_raw is not None else ''
    acquisition_part = ('-' + str(acquisition_raw)) if acquisition_raw is not None else ''

    name = toolsversion_str + '-' + datatype + dataset_part + acquisition_part
    return find_filename_without_duplicates(save_directory, name, extension='npz')


def set_index(index: Dict[str, Any]):
    """
    Overwrite the on-disk index with `index`.

    Parameters
    ----------
    index : dict
        Mapping of filename → parameters dict.  The whole dict is stored as
        a single pickled object inside an ``.npz`` archive.

    Notes
    -----
    Writes to the module-level `index_path`.  Any previous content is
    silently replaced.
    """
    ensure_save_directory_exists()
    np.savez(index_path, index=index)


def clean_index(index: Dict[str, Any]):
    """
    Remove entries from `index` whose backing files no longer exist on disk.

    Mutates `index` in place **and** persists the cleaned version via
    `set_index`.

    Parameters
    ----------
    index : dict
        Mapping of filename → parameters dict, as returned by `get_index`.
        Modified in place.

    Notes
    -----
    Entries are removed from the dict; the corresponding files, if somehow
    still present despite being flagged as missing, are also deleted.  In
    practice that second branch should never trigger (the item was added
    because the file was absent), but it is kept as a safety net.
    """
    bad_items = [
        item for item in index.keys()
        if not os.path.isfile(os.path.join(save_directory, item))
    ]
    customlog.log_debug(f'clean_index: {len(bad_items)} stale entry/entries found')

    for item in bad_items:
        if item not in index:
            continue  # already removed by a previous iteration (shouldn't happen)
        customlog.log_info(
            f'Removing item {item} from index since it no longer exists '
            f'(did you delete it manually?)'
        )
        index.pop(item)
        customlog.log_debug(f'Removed item {item} from index')

        # NOTE: this branch is effectively dead — the item was listed because
        # the file was absent — but kept as a defensive measure.
        file_path = os.path.join(save_directory, item)
        if os.path.isfile(file_path):
            os.remove(file_path)
            customlog.log_debug(f'Deleted orphan file {file_path}')

    set_index(index)


def get_index() -> Dict[str, Any]:
    """
    Load the on-disk index, cleaning stale entries before returning it.

    Creates an empty index file if none exists yet.

    Returns
    -------
    dict
        Mapping of filename → parameters dict for every currently valid
        cached file.
    """
    ensure_save_directory_exists()
    if not os.path.isfile(index_path):
        customlog.log_debug('No index file found — creating an empty one')
        np.savez(index_path, index={})
    index = np.load(index_path, allow_pickle=True)['index'][()]
    customlog.log_trace(f'get_index: loaded {len(index)} raw entry/entries before cleaning')
    clean_index(index)
    customlog.log_trace(f'get_index: {len(index)} entry/entries after cleaning')
    return index


def get_items(parameters: dict, total_match: bool = False) -> List[str]:
    """
    Return filenames whose stored parameters match `parameters`.

    Parameters
    ----------
    parameters : dict
        Query parameters.  The keys ``'framenumbers'`` and ``'verbose'`` are
        always ignored during matching (they are considered transient).
    total_match : bool, optional
        If ``True``, the candidate's key set must equal `parameters`'s key
        set exactly (after ignoring the transient keys).
        If ``False`` (default), every key in `parameters` (except transient
        ones) must be present and equal in the candidate.

    Returns
    -------
    list of str
        Filenames (no directory prefix) of matching cached files, in
        index-iteration order.

    Notes
    -----
    Value comparison uses ``np.prod(candidate == query)`` so that numpy
    arrays are compared element-wise and scalars work uniformly.
    """
    IGNORED_KEYS = {'framenumbers', 'verbose'}

    index = get_index()
    items = []
    customlog.log_subtrace(f'get_items: searching index ({len(index)} entries) for {parameters}')

    query_keys = [k for k in parameters.keys() if k not in IGNORED_KEYS]

    for candidate_file_name, candidate_parameters in index.items():
        # --- key-set check ---
        if total_match:
            candidate_keys = [k for k in candidate_parameters.keys() if k not in IGNORED_KEYS]
            query_keys_set = set(query_keys)
            if set(candidate_keys) != query_keys_set:
                continue
        else:
            if not all(k in candidate_parameters for k in query_keys):
                continue

        # --- value check ---
        matches = []
        for key in query_keys:
            result = np.prod(candidate_parameters[key] == parameters[key])
            # Collapse any remaining array dims to a scalar bool
            while np.asarray(result).shape != ():
                result = np.prod(result)
            matches.append(bool(result))
            if not bool(result):
                customlog.log_subtrace(
                    f'get_items: key "{key}" mismatch in "{candidate_file_name}" '
                    f'(stored={candidate_parameters[key]!r}, query={parameters[key]!r})'
                )

        if all(matches):
            customlog.log_subtrace(f'get_items: match → "{candidate_file_name}"')
            items.append(candidate_file_name)

    customlog.log_trace(
        f'get_items: found {len(items)} item(s) with '
        f'{"total" if total_match else "partial"} match'
    )
    customlog.log_subtrace(f'get_items: matched files = {items}')
    return items


def framenumbers_available(parameters: dict) -> Optional[np.ndarray]:
    """
    Return the framenumbers already cached for the given parameters.

    Parameters
    ----------
    parameters : dict
        Must contain ``'acquisition'`` and ``'dataset'`` for logging.
        Other keys are forwarded to `get_items`.

    Returns
    -------
    numpy.ndarray or None
        * ``None``  — a matching file exists but stores no ``'framenumbers'``
          key (interpreted as *all* frames available).
        * Empty array — no matching file found.
        * Non-empty array — the framenumbers stored in the first matching
          file.

    Notes
    -----
    Only the first matching item is used.  If multiple matches exist the
    extras are silently ignored.
    """
    acq = parameters.get('acquisition', '???')
    ds = parameters.get('dataset', '???')
    customlog.log_debug(f'framenumbers_available: checking cache for {acq} ({ds})')

    items = get_items(parameters, total_match=False)
    if not items:
        customlog.log_debug('framenumbers_available: no matching item found')
        return np.empty(0, int)

    index = get_index()
    for item in items:
        file_path = os.path.join(save_directory, item)
        if os.path.isfile(file_path):
            framenumbers = index[item].get('framenumbers', None)
            customlog.log_trace(
                f'framenumbers_available: "{item}" → '
                f'{"all frames" if framenumbers is None else f"{len(framenumbers)} frame(s)"}'
            )
            return framenumbers
        else:
            customlog.log_warning(f'framenumbers_available: index entry "{item}" has no file ?!')

    return np.empty(0, int)


def fetch_saved_data(parameters: dict) -> Any:
    """
    Load and return the first cached data array matching `parameters`.

    Parameters
    ----------
    parameters : dict
        Query parameters forwarded to `get_items`.  ``'datatype'``,
        ``'acquisition'``, and ``'dataset'`` are used for log messages.

    Returns
    -------
    Any
        The ``'data'`` payload from the ``.npz`` file, or ``None`` if no
        match is found.

    Notes
    -----
    Only the first matching file is loaded.
    """
    dtype = parameters.get('datatype', 'data')
    acq = parameters.get('acquisition', '???')
    ds = parameters.get('dataset', '???')
    customlog.log_debug(f'fetch_saved_data: fetching "{dtype}" for {acq} ({ds})')

    items = get_items(parameters, total_match=False)
    for item in items:
        file_path = os.path.join(save_directory, item)
        if os.path.isfile(file_path):
            customlog.log_trace(f'fetch_saved_data: loading "{item}"')
            return np.load(file_path, allow_pickle=True)['data'][()]

    customlog.log_warning('fetch_saved_data: no matching file found')
    return None


def erase_items(parameters: dict, total_match: bool = False) -> None:
    """
    Delete all cached files (and their index entries) matching `parameters`.

    Parameters
    ----------
    parameters : dict
        Query parameters forwarded to `get_items`.  If ``None``, the
        function returns immediately.
    total_match : bool, optional
        Forwarded to `get_items` (default ``False``).

    Returns
    -------
    None
    """
    if parameters is None:
        return None

    customlog.log_trace('erase_items: starting')
    index = get_index()
    items = get_items(parameters=parameters, total_match=total_match)
    customlog.log_debug(f'erase_items: {len(items)} item(s) to remove')

    for item in items:
        if item not in index:
            continue
        index.pop(item)
        customlog.log_trace(f'erase_items: removed "{item}" from index')
        file_path = os.path.join(save_directory, item)
        if os.path.isfile(file_path):
            os.remove(file_path)
            customlog.log_trace(f'erase_items: deleted file "{file_path}"')
        else:
            customlog.log_warning(f'erase_items: expected file "{file_path}" not found on disk')

    set_index(index)


def erase_all() -> None:
    """
    Erase all cached data.

    .. warning::
        Not implemented.  Raises ``NotImplementedError``.
    """
    # SANITY: the original silently logged an error; raising is safer so
    # callers aren't left thinking the operation succeeded.
    raise NotImplementedError('erase_all() is not yet implemented')


def nuke_all() -> None:
    """
    Irrevocably delete every file in the save directory and the directory itself.

    .. danger::
        This cannot be undone.  All cached data and the index are permanently
        removed.

    Returns
    -------
    None
    """
    customlog.log_info(f'nuke_all: deleting everything in "{save_directory}"')
    for file in os.listdir(save_directory):
        full_path = os.path.join(save_directory, file)
        if os.path.isfile(full_path):
            os.remove(full_path)
            customlog.log_trace(f'nuke_all: deleted "{full_path}"')
    os.rmdir(save_directory)
    customlog.log_info('nuke_all: done')


def add_item_to_index(item: str, parameters: dict,
                      remove_similar_older_entries: bool = False) -> None:
    """
    Register a file in the index under its generating parameters.

    Parameters
    ----------
    item : str
        Filename (no directory prefix) of the file to register.
    parameters : dict
        Parameters that generated the file.  If ``None``, the function
        returns immediately.
    remove_similar_older_entries : bool, optional
        If ``True``, calls `erase_items` with a partial match before adding
        the new entry, preventing accumulation of stale files (default
        ``False``).

    Returns
    -------
    None
    """
    if parameters is None:
        return None

    # SANITY: item should be a bare filename, not a full path
    if os.sep in item:
        customlog.log_warning(
            f'add_item_to_index: "item" contains a path separator — '
            f'expected a bare filename, got "{item}"'
        )

    if remove_similar_older_entries:
        customlog.log_trace(f'add_item_to_index: erasing similar older entries before adding "{item}"')
        erase_items(parameters=parameters, total_match=False)

    index = get_index()
    index[item] = parameters
    set_index(index)
    customlog.log_trace(f'add_item_to_index: registered "{item}"')


def save_data(data: Any, parameters: dict) -> None:
    """
    Persist `data` to disk and register it in the index.

    Any previously cached file with partially matching parameters is replaced
    (via `add_item_to_index` with ``remove_similar_older_entries=True``).

    Parameters
    ----------
    data : Any
        Data to save.  Must be serialisable by ``numpy.savez``.  If ``None``
        the function logs a warning and returns without writing.
    parameters : dict
        Parameters that produced `data`.  If ``None`` the function logs a
        warning and returns without writing.

    Returns
    -------
    None
    """
    if parameters is None:
        customlog.log_warning('save_data: parameters are None — skipping save')
        return None
    if data is None:
        customlog.log_warning('save_data: data is None — skipping save')
        return None

    customlog.log_trace('save_data: saving item')
    filename = generate_appropriate_filename(parameters)
    add_item_to_index(filename, parameters, remove_similar_older_entries=True)
    np.savez(os.path.join(save_directory, filename), data=data)
    customlog.log_trace(f'save_data: saved "{filename}"')


def auto_data_generating_fn(parameters: Dict[str, Any]) -> Any:
    """
    Dispatch data generation to the appropriate rivulet-finding routine.

    Parameters
    ----------
    parameters : dict
        Must contain ``'datatype'``.  Additional keys are forwarded as
        keyword arguments to the underlying routine.  ``'dataset'`` and
        ``'acquisition'`` should normally be present; a warning is logged
        when they are missing.

    Returns
    -------
    Any
        The generated data, or ``None`` if ``datatype`` is ``None`` or
        unrecognised.
    """
    datatype = parameters.get('datatype', None)
    customlog.log_debug(f'auto_data_generating_fn: generating "{datatype}"')

    # SANITY: warn if commonly required keys are absent
    for required_key in ('dataset', 'acquisition'):
        if required_key not in parameters:
            customlog.log_warning(
                f'auto_data_generating_fn: "{required_key}" not found in parameters — '
                f'downstream call may fail'
            )

    if datatype is None:
        customlog.log_error('auto_data_generating_fn: datatype is None')
        return None
    elif datatype == 'bos':
        return rivuletfinding.find_bos(**parameters)
    elif datatype == 'borders':
        return rivuletfinding.find_borders(**parameters)
    elif datatype == 'fwhmol':
        return rivuletfinding.find_fwhmol(**parameters)
    elif datatype == 'bol':
        return rivuletfinding.find_bol(**parameters)
    else:
        customlog.log_error(f'auto_data_generating_fn: unrecognised datatype "{datatype}"')
        return None


def fetch_or_generate(datatype: Optional[str] = None, dataset: Optional[str] = None, acquisition: Optional[str] = None,
                      parameters: dict = None, data_generating_fn=None) -> Any:
    """
    Convenience wrapper around `fetch_or_generate_data_from_parameters`.

    Builds a parameters dict from the explicit ``dataset`` and ``acquisition``
    arguments, merges any extra entries from ``parameters``, then delegates to
    `fetch_or_generate_data_from_parameters`.

    Parameters
    ----------
    datatype : str
        Type of data to fetch or generate (e.g. ``'bos'``, ``'borders'``).
    dataset : str
        Dataset identifier.
    acquisition : str
        Acquisition identifier.
    parameters : dict, optional
        Additional parameters merged into the call (default ``None``, treated
        as an empty dict).  ``'dataset'`` and ``'acquisition'`` keys in this
        dict are overridden by the explicit arguments.
    data_generating_fn : callable, optional
        Custom generation function; forwarded verbatim.

    Returns
    -------
    Any
        Cached or freshly generated data.
    """
    # SANITY: parameters=None would crash on **parameters
    extra = {** parameters} if parameters is not None else {}
    if acquisition is not None:
        extra['acquisition'] = acquisition
    if dataset is not None:
        extra['dataset'] = dataset
    return fetch_or_generate_data_from_parameters(datatype, extra,
                                                  data_generating_fn=data_generating_fn)


def fetch_or_generate_data_from_parameters(datatype: str, parameters: dict,
                                            data_generating_fn=None) -> Any:
    """
    Return data for `datatype`, using the cache when possible.

    The function operates in one of two modes depending on whether
    ``'framenumbers'`` is present as a key in `parameters`:

    **Generic mode** (``'framenumbers'`` key absent)
        A simple fetch-or-generate cycle: if a matching entry exists in the
        cache it is returned immediately; otherwise `data_generating_fn` is
        called once, the result is saved, and returned.  No chunking is
        performed.

    **Video mode** (``'framenumbers'`` key present)
        Checks which framenumbers are already cached.  If the wanted subset is
        fully covered, the cached slice is returned.  Otherwise the missing
        frames are generated in chunks of 500, the complete array is saved,
        and the originally requested subset is returned via a single
        tail-recursion.  ``'framenumbers'`` may be ``None`` (meaning *all
        available frames*) or an integer array of specific frame indices.

    Parameters
    ----------
    datatype : str
        Type of data (e.g. ``'bos'``, ``'borders'``, ``'fwhmol'``,
        ``'bol'``).
    parameters : dict
        Generation / retrieval parameters.  ``'datatype'`` is injected
        automatically.  In video mode the ``'framenumbers'`` key is
        read and may be mutated during processing; pass a copy if the
        caller's dict must stay unchanged.
    data_generating_fn : callable, optional
        Function with signature ``fn(parameters: dict) -> Any`` used to
        produce data when the cache misses.  Defaults to
        `auto_data_generating_fn` when ``None``.

    Returns
    -------
    Any
        Cached or freshly generated data, or ``None`` on unrecoverable error.

    Notes
    -----
    In video mode, when ``available_fns`` is ``None`` (all frames cached) and
    ``wanted_fns`` is a specific array, ``data[wanted_fns]`` is returned
    without shape-compatibility validation — the caller is responsible for
    ensuring the index is meaningful for the stored array.

    The ``'bol'`` and ``'fwhmol'`` datatypes unconditionally pre-generate
    ``'borders'`` as a side-effect (video mode only, via the rivulet
    workflow).
    """
    customlog.log_subinfo(f'fetch_or_generate_data_from_parameters: "{datatype}"')
    customlog.log_subtrace(f'Parameters: {parameters}')

    parameters['datatype'] = datatype

    if data_generating_fn is None:
        data_generating_fn = auto_data_generating_fn

    # ------------------------------------------------------------------ #
    # GENERIC MODE — no framenumbers key in parameters                    #
    # ------------------------------------------------------------------ #
    if 'framenumbers' not in parameters:
        customlog.log_debug('fetch_or_generate_data_from_parameters: generic mode (no framenumbers)')

        cached = fetch_saved_data(parameters)
        if cached is not None:
            customlog.log_debug('fetch_or_generate_data_from_parameters: cache hit — returning cached data')
            return cached

        customlog.log_debug('fetch_or_generate_data_from_parameters: cache miss — generating data')
        data = data_generating_fn(parameters)
        if data is None:
            customlog.log_error('fetch_or_generate_data_from_parameters: generation returned None')
            return None
        save_data(data, parameters)
        return data

    # ------------------------------------------------------------------ #
    # VIDEO MODE — framenumbers key is present                            #
    # ------------------------------------------------------------------ #
    customlog.log_debug('fetch_or_generate_data_from_parameters: video mode (framenumbers present)')

    # Pre-generate border data for derived types (rivulet-specific)
    if datatype in ('bol', 'fwhmol'):
        customlog.log_debug(f'Pre-generating "borders" required by "{datatype}"')
        fetch_or_generate_data_from_parameters('borders', {**parameters},
                                               data_generating_fn=data_generating_fn)

    available_fns = framenumbers_available(parameters)
    wanted_fns = parameters['framenumbers']
    customlog.log_trace(f'wanted_framenumbers    = {wanted_fns}')
    customlog.log_trace(f'available_framenumbers = {available_fns}')

    if available_fns is None:
        # All frames are cached
        customlog.log_debug('All framenumbers available in cache')
        data = fetch_saved_data(parameters)

        if data is None:
            # Cache reported everything available but fetch failed — fall through to regeneration
            customlog.log_warning(
                'fetch_or_generate_data_from_parameters: '
                'cache indicated all fns available but fetch returned None — regenerating'
            )
        else:
            return data if wanted_fns is None else data[wanted_fns]

    else:
        # Partial or no cache
        if wanted_fns is not None:
            overlap = np.isin(wanted_fns, available_fns, assume_unique=True)
            customlog.log_debug(
                f'{len(available_fns)} fns available, '
                f'{len(wanted_fns)} wanted, '
                f'{overlap.sum()} overlap'
            )
            if overlap.sum() == len(wanted_fns):
                # All wanted frames are cached — return the slice directly
                customlog.log_debug('All wanted framenumbers cached — skipping generation')
                data = fetch_saved_data(parameters)
                return data[np.isin(available_fns, wanted_fns, assume_unique=True)]
            else:
                # Some wanted frames are missing — generate the union
                total_fns = np.sort(np.unique(np.concatenate((wanted_fns, available_fns))))
                customlog.log_debug(
                    f'Need to generate {len(total_fns) - len(available_fns)} new frame(s)'
                )
        else:
            customlog.log_debug('All framenumbers wanted; will generate everything')
            total_fns = None

    # --- Generation (video mode) ---
    customlog.log_debug('Generating video data')
    total_fns_explicit = total_fns
    if total_fns is None:
        acquisition_path = rivuletfinding.get_acquisition_path_from_parameters(**parameters)
        total_fns_explicit = np.arange(
            videoreading.get_number_of_available_frames(acquisition_path)
        )
        customlog.log_debug(f'Total frames in acquisition: {len(total_fns_explicit)}')

    if len(total_fns_explicit) == 0:
        customlog.log_warning('fetch_or_generate_data_from_parameters: no frames to generate')
        return None

    chunk_size: int = 500
    number_of_chunks: int = (
        len(total_fns_explicit) // chunk_size
        + (len(total_fns_explicit) % chunk_size != 0)
    )
    customlog.log_debug(
        f'Splitting {len(total_fns_explicit)} frame(s) into '
        f'{number_of_chunks} chunk(s) of {chunk_size}'
    )

    # First chunk
    parameters['framenumbers'] = total_fns_explicit[:chunk_size]
    customlog.log_info(f'Generating {datatype} (chunk 1/{number_of_chunks})')
    data = data_generating_fn(parameters)

    if data is None:
        customlog.log_error(
            'fetch_or_generate_data_from_parameters: '
            'first chunk returned None — aborting generation'
        )
        return None

    # Subsequent chunks
    for i in range(1, number_of_chunks):
        customlog.log_info(f'Generating {datatype} (chunk {i + 1}/{number_of_chunks})')
        parameters['framenumbers'] = total_fns_explicit[i * chunk_size:(i + 1) * chunk_size]
        chunk = data_generating_fn(parameters)

        if chunk is None:
            customlog.log_warning(
                f'fetch_or_generate_data_from_parameters: '
                f'chunk {i + 1}/{number_of_chunks} returned None — skipping'
            )
            continue

        data = np.concatenate((data, chunk))

    # Save the full generated array, then recurse once to return the right subset
    parameters['framenumbers'] = total_fns
    save_data(data, parameters)

    parameters['framenumbers'] = wanted_fns
    return fetch_or_generate_data_from_parameters(datatype, parameters,
                                                  data_generating_fn=data_generating_fn)