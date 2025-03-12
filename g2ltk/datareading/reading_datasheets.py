from typing import Optional, Any, Tuple, Dict, List, Union
import os # to navigate in the directories
import numpy as np
import pandas as pd

from .. import logging

from . import generate_dataset_path, find_available_videos

# a worksheet is part of an excel workbook
__dataworkbook_name_default__ = 'dataworkbook'
__dataworkbook_metasheet_name_default__ = 'metainfo'
__dataworkbook_acquisition_title_column_name_default__ = 'acquisition_title'

def generate_dataworkbook_path(dataset:str) -> Optional[str]:
    dataset_path = generate_dataset_path(dataset)

    ext = '.xlsx'
    xlsx_spreadsheets = [f[:-len(ext)] for f in os.listdir(dataset_path) if os.path.isfile(os.path.join(dataset_path, f)) and f.endswith(ext)]
    xlsx_spreadsheets.sort()

    ext = '.ods'
    ods_spreadsheets = [f[:-len(ext)] for f in os.listdir(dataset_path) if os.path.isfile(os.path.join(dataset_path, f)) and f.endswith(ext)]
    ods_spreadsheets.sort()

    datasheet_name = None

    # First, we try the default name
    global __dataworkbook_name_default__
    if os.path.isfile(os.path.join(dataset_path, __dataworkbook_name_default__ + '.xlsx')):
        datasheet_name = __dataworkbook_name_default__ + '.xlsx'
    elif os.path.isfile(os.path.join(dataset_path, __dataworkbook_name_default__ + '.ods')):
        logging.log_warning('Old format .ods used, consider upgrading to .xlsx')
        datasheet_name = __dataworkbook_name_default__ + '.ods'
    elif len(xlsx_spreadsheets) == 1 and len(ods_spreadsheets) == 0:
        datasheet_name = xlsx_spreadsheets[0] + '.xlsx'
        logging.log_warning(f'Name of datasheet is "{datasheet_name}", consider changing to "{__dataworkbook_name_default__ + ".xlsx"}" for consistency')
    elif len(xlsx_spreadsheets) == 0 and len(ods_spreadsheets) == 1:
        datasheet_name = ods_spreadsheets[0] + '.ods'
        logging.log_warning(f'Name of datasheet is "{datasheet_name}", consider changing to "{__dataworkbook_name_default__ + ".xlsx"}" for consistency')
    elif len(xlsx_spreadsheets) == 0 and len(ods_spreadsheets) == 0:
        logging.log_warning(f'No datasheet in dataset {dataset}.')
    else:
        logging.log_warning(f'To many datasheet in dataset {dataset}: {xlsx_spreadsheets+ods_spreadsheets}.')
    if datasheet_name is None:
        return None
    datasheet_path = os.path.join(dataset_path, datasheet_name)
    return datasheet_path

def obtain_metainfo(dataset:str) -> Optional[Any]:
    datasheet_path = generate_dataworkbook_path(dataset)
    if datasheet_path is None:
        return None

    datasheet_file = pd.ExcelFile(datasheet_path)
    global __dataworkbook_metasheet_name_default__
    if __dataworkbook_metasheet_name_default__ in datasheet_file.sheet_names:
        metasheet_name = __dataworkbook_metasheet_name_default__
    else:
        metasheet_name = datasheet_file.sheet_names[0]
        logging.log_warning(f'Name of the metainfo sheet is {metasheet_name}, consider changing to "{__dataworkbook_metasheet_name_default__}" for consistency.')

    metainfo = pd.read_excel(datasheet_file, sheet_name=metasheet_name, skiprows=2)
    return metainfo

def obtain_acquisition_titles(dataset:str) -> Optional[np.ndarray]:
    metainfo = obtain_metainfo(dataset)
    if metainfo is None:
        return None

    meaningful_keys = [key for key in metainfo.keys() if 'unnamed' not in key.lower()]

    acquisition_title_key = __dataworkbook_acquisition_title_column_name_default__
    if not __dataworkbook_acquisition_title_column_name_default__ in meaningful_keys:
        if 'title' in meaningful_keys:
            acquisition_title_key = 'title'
            logging.log_warning(f"Name of the acquisition_title column is '{acquisition_title_key}', consider changing to '{__dataworkbook_acquisition_title_column_name_default__}' for consistency.")
        else:
            logging.log_error(f"Did not find the column corresponding to 'acquisition_title' in the datasheet")
            return None
    valid = metainfo[acquisition_title_key].astype(str) != 'nan'

    acquisition_titles:np.ndarray = metainfo[acquisition_title_key][valid].to_numpy()

    all_videos:List[str] = find_available_videos(dataset=dataset)

    for real_video in all_videos:
        if real_video not in acquisition_titles:
            if not real_video.endswith('_gcv'):
                logging.log_warning(f"Video '{real_video}' seems to exist but not be recorded in the datasheet")

    for listed_video in acquisition_titles:
        if listed_video not in all_videos:
            logging.log_warning(f"Video '{listed_video}' is listed in the datasheet but does not seem to exit.")

    return acquisition_titles

def load_or_create_worksheet(dataset:str, sheet_name:str, columns:Optional[List[str]]=None) -> Optional[pd.DataFrame]:
    workbook_path = generate_dataworkbook_path(dataset)
    if workbook_path is None:
        logging.log_error(f'Did not found datasheet for dataset {dataset}.')
        return None

    datasheet_file = pd.ExcelFile(workbook_path)
    if sheet_name in datasheet_file.sheet_names:
        logging.log_debug(f'Found worksheet {sheet_name} for dataset {dataset}.')
        datasheet:pd.DataFrame = pd.read_excel(workbook_path, sheet_name=sheet_name)
        existing_sheet_keys = [key for key in datasheet.keys()]
        logging.log_debug(f'Existing sheet columns: {existing_sheet_keys}.')
        if columns is not None:
            logging.log_debug(f'Wanted columns: {columns}.')
            unexisting_wanted_key = []
            for wanted_key in columns:
                if wanted_key not in existing_sheet_keys:
                    unexisting_wanted_key.append(wanted_key)
                    logging.log_warning(f"Worksheet {sheet_name} does not contain column {wanted_key}.")
            if len(unexisting_wanted_key) > 0:
                    # # Old behaviour : just fail.
                    # logging.log_error(f"Operation impossible: Worksheet {sheet_name} does not contain columns {unexisting_wanted_key}")
                    # return None

                    # New behaviour : we add the missing columns
                    acquisition_titles = obtain_acquisition_titles(dataset)
                    for wanted_key in unexisting_wanted_key:
                        datasheet[wanted_key] = np.full(len(acquisition_titles), np.nan)
                    overwrite_datasheet(dataset, datasheet, sheet_name=sheet_name)
                    logging.log_warning(f"Added columns {unexisting_wanted_key} to worksheet {sheet_name}.")
    else:
        # We must create the sheet ourselves
        acquisition_titles = obtain_acquisition_titles(dataset)
        datadict = {__dataworkbook_acquisition_title_column_name_default__: acquisition_titles}
        for column in columns:
            datadict[column] = np.full(len(acquisition_titles), np.nan)

        datasheet = pd.DataFrame(datadict)

        with pd.ExcelWriter(workbook_path, engine='openpyxl', mode='a', if_sheet_exists='error') as writer:
            datasheet.to_excel(writer, sheet_name=sheet_name, index=False)
        logging.log_info(f'Created worksheet named {sheet_name} for dataset {dataset}.')
        logging.log_debug(f'worksheet {sheet_name} has columns {columns}')

    return pd.read_excel(workbook_path, sheet_name=sheet_name)


def overwrite_datasheet(dataset:str, datasheet:pd.DataFrame, sheet_name:str):
    workbook_path = generate_dataworkbook_path(dataset)
    if workbook_path is None:
        logging.log_error(f'Did not found datasheet for dataset {dataset}.')
        return None

    with pd.ExcelWriter(workbook_path, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
        datasheet.to_excel(writer, sheet_name=sheet_name, index=False)




































